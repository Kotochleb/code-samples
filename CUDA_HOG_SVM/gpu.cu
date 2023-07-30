#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cuda.h"
#include "ppmIO.h"
#include <math.h>
#include <stdexcept>
#include "svm.h"
#include <chrono>
#include <iostream>

using namespace std::chrono;

#define SUCCESS 0
#define NO_FILE 1

#define TILE_SIZE 8
#define NUM_BINS 9
#define NUM_CELLS_IN_BLOCKS 4
#define NUM_CHAN 3

// #define OUT_TO_HOST

// Global index
#define GI_IDX(c, w, h) (x_idx(NUM_CHAN, height, width, (c), (h), (w)))
// Shared index
#define SI_IDX(c, w, h) (x_idx(NUM_CHAN, TILE_SIZE + 2, TILE_SIZE + 2, (c), (h), (w)))
#define ST_IDX(w, h) (x_idx(NUM_CHAN, TILE_SIZE, TILE_SIZE, (1), (h - 1), (w - 1)))
#define HIST_IDX(b, tw, th) (x_idx(NUM_BINS + 1, grid_y, grid_x, (b), (th), (tw)))

__device__ int x_idx(int C, int H, int W, int c, int h, int w)
{
    return (h * (C * W)) + (w * (C)) + (c * (1));
}

#ifdef OUT_TO_HOST
__global__ void HOG_kernel(int width, int height, float *img, float *hist, float *GX, float *GY)
#endif
#ifndef OUT_TO_HOST
    __global__ void histogram_cell_kernel(int width, int height, float *img, float *hist)
#endif
{
    extern __shared__ uint8_t shm[];

    const size_t tile_full_size = NUM_CHAN * (TILE_SIZE + 2) * (TILE_SIZE + 2) * sizeof(float);
    size_t max_grad_idx_tile_size = TILE_SIZE * TILE_SIZE * sizeof(uint8_t);
    float *s_X = (float *)&(shm[0]);
    uint8_t *s_max_idx = (uint8_t *)&(shm[tile_full_size]);
    float *s_hist = (float *)&(shm[tile_full_size + max_grad_idx_tile_size]);

    size_t grid_x = floor((float)width / (float)TILE_SIZE);
    size_t grid_y = floor((float)height / (float)TILE_SIZE);

    // Local tile coordiantes
    unsigned int x = threadIdx.x + 1;
    unsigned int y = threadIdx.y + 1;
    unsigned int c = threadIdx.z; // channel

    // tile x and y
    unsigned int tx = blockIdx.x;
    unsigned int ty = blockIdx.y;

    // Global image coordinates
    unsigned int gx = tx * TILE_SIZE + x - 1;
    unsigned int gy = ty * TILE_SIZE + y - 1;

    // Load data to tile
    s_X[SI_IDX(c, x, y)] = img[GI_IDX(c, gx, gy)];

    // Corner cases... Yay...
    if (x == 1)
    {
        if (gx > 0)
        {
            s_X[SI_IDX(c, x - 1, y)] = img[GI_IDX(c, gx - 1, gy)];
        }
        else
        {
            s_X[SI_IDX(c, x - 1, y)] = img[GI_IDX(c, gx, gy)];
        }
    }
    else if (x == TILE_SIZE)
    {
        if (gx < width - 1)
        {
            s_X[SI_IDX(c, x + 1, y)] = img[GI_IDX(c, gx + 1, gy)];
        }
        else
        {
            s_X[SI_IDX(c, x + 1, y)] = img[GI_IDX(c, gx, gy)];
        }
    }

    if (y == 1)
    {
        if (gy > 0)
        {
            s_X[SI_IDX(c, x, y - 1)] = img[GI_IDX(c, gx, gy - 1)];
        }
        else
        {
            s_X[SI_IDX(c, x, y - 1)] = img[GI_IDX(c, gx, gy)];
        }
    }
    else if (y == TILE_SIZE)
    {
        if (gy < height - 1)
        {
            s_X[SI_IDX(c, x, y + 1)] = img[GI_IDX(c, gx, gy + 1)];
        }
        else
        {
            s_X[SI_IDX(c, x, y + 1)] = img[GI_IDX(c, gx, gy)];
        }
    }

    __syncthreads();

    // Compute gradients
    float grad_x = s_X[SI_IDX(c, x + 1, y)] - s_X[SI_IDX(c, x - 1, y)];
    float grad_y = s_X[SI_IDX(c, x, y + 1)] - s_X[SI_IDX(c, x, y - 1)];

    // save gradient value to shared memory
    s_X[SI_IDX(c, x, y)] = sqrt((grad_x * grad_x) + (grad_y * grad_y));

#ifdef OUT_TO_HOST
    GY[GI_IDX(c, gx, gy)] = s_X[SI_IDX(c, x, y)];
#endif

    __syncthreads();

    // find maximum
    if (c == 0)
    {
        s_max_idx[ST_IDX(x, y)] = 0;
        float max_amp = s_X[SI_IDX(0, x, y)];
        for (size_t i = 1; i < NUM_CHAN; i++)
        {
            if (s_X[SI_IDX(i, x, y)] > max_amp)
            {
                s_max_idx[ST_IDX(x, y)] = i;
                max_amp = s_X[SI_IDX(i, x, y)];
            }
        }
    }

    __syncthreads();

#ifdef OUT_TO_HOST
    GX[GI_IDX(c, gx, gy)] = s_X[SI_IDX(s_max_idx[ST_IDX(x, y)], x, y)];
#endif

    unsigned int hidx = threadIdx.x * TILE_SIZE + threadIdx.y;
    // Ensure histogram is full of zeros
    if (hidx <= NUM_BINS)
    {
        s_hist[HIST_IDX(hidx, tx, ty)] = 0.0f;
    }

    __syncthreads();

    // compute histogram
    if (c == s_max_idx[ST_IDX(x, y)])
    {
        float amp = s_X[SI_IDX(c, x, y)];
        float theta = atan2f(grad_y, grad_x) / M_PI;
        theta = theta >= 0.0f ? theta : theta + 1.0;
        theta *= (float)(NUM_BINS - 1);
        float low = floor(theta);
        float up = ceilf(theta);

        atomicAdd(&(s_hist[HIST_IDX((int)low, tx, ty)]), (theta - low) * amp);
        atomicAdd(&(s_hist[HIST_IDX((int)up, tx, ty)]), (up - theta) * amp);
    }

    if (c == 0 && hidx < NUM_BINS)
    {
        // Cooperate loading to global memory
        hist[HIST_IDX(hidx, tx, ty)] = s_hist[HIST_IDX(hidx, tx, ty)];
    }
}

#define NORM_HIST_IDX(gcx, gcy, c, b) (norm_hist_idx(gridDim.y, NUM_CELLS_IN_BLOCKS, NUM_BINS, (gcx), (gcy), (c), (b)))
__device__ int norm_hist_idx(int CY, int C, int B, int cx, int cy, int c, int b)
{
    return (cx * (CY * C * B)) + (cy * (C * B)) + (c * (B)) + (b * (1));
}

__global__ void histogram_block_kernel(float *hist, float *norm_hist)
{
    // local cell number
    unsigned int c = threadIdx.x;
    unsigned int cx = threadIdx.x / 2;
    unsigned int cy = threadIdx.x % 2;
    // bin number
    unsigned int b = threadIdx.y;

    // global cell index in x and y
    unsigned int gcx = blockIdx.x;
    unsigned int gcy = blockIdx.y;

    // // used by HIST_IDX macro
    size_t grid_x = gridDim.x + 1;
    size_t grid_y = gridDim.y + 1;

    size_t norm_his_size = blockDim.x * NUM_BINS * sizeof(float);

    extern __shared__ uint8_t shm[];
    float *s_norm_hist = (float *)&(shm[0]);
    float *s_accumulator = (float *)&(shm[norm_his_size]);

    size_t flat_idx = c * NUM_BINS + b;

    // cooperate loading histograms
    s_norm_hist[flat_idx] = hist[HIST_IDX(b, gcx + cx, gcy + cy)];

    __syncthreads();

    // perform reduction
    size_t flat_array_len = NUM_CELLS_IN_BLOCKS * NUM_BINS;
    size_t last_i_passing = 0;

    // rewrite values to accumulator as sum of (Austin) Powers
    if (flat_idx < flat_array_len / 2)
    {
        float a = s_norm_hist[flat_idx];
        float b = s_norm_hist[flat_idx + flat_array_len / 2];
        s_accumulator[flat_idx] = a * a + b * b;
    }
    for (size_t i = 1; i < flat_array_len; i *= 2)
    {
        size_t thread_idx_lim = flat_array_len / i;
        if (flat_idx < thread_idx_lim)
        {
            float a = s_accumulator[flat_idx];
            float b = s_accumulator[flat_idx + thread_idx_lim];
            s_accumulator[flat_idx] = a + b;
        }
        last_i_passing = i;
        __syncthreads();
    }

    // Add reamining values that can not be added using reduction
    if (flat_idx == 0)
    {
        for (size_t i = last_i_passing / 2; i < flat_array_len / 2; i++)
        {
            s_accumulator[flat_idx] += s_accumulator[i];
        }
        s_accumulator[flat_idx] = sqrt(s_accumulator[flat_idx] + 1e-4);
    }

    __syncthreads();

    // L2 norm f1 equation
    float f1 = s_norm_hist[flat_idx] / s_accumulator[0];
    // threshold
    s_norm_hist[flat_idx] = fminf(0.2, f1);

    __syncthreads();

    // rewrite values to accumulator as sum of (Austin) Powers
    if (flat_idx < flat_array_len / 2)
    {
        float a = s_norm_hist[flat_idx];
        float b = s_norm_hist[flat_idx + flat_array_len / 2];
        s_accumulator[flat_idx] = a * a + b * b;
    }
    for (size_t i = 1; i < flat_array_len; i *= 2)
    {
        size_t thread_idx_lim = flat_array_len / i;
        if (flat_idx < thread_idx_lim)
        {
            float a = s_accumulator[flat_idx];
            float b = s_accumulator[flat_idx + thread_idx_lim];
            s_accumulator[flat_idx] = a + b;
        }
        last_i_passing = i;
        __syncthreads();
    }

    // Add reamining values that can not be added using reduction
    if (flat_idx == 0)
    {
        for (size_t i = last_i_passing / 2; i < flat_array_len / 2; i++)
        {
            s_accumulator[flat_idx] += s_accumulator[i];
        }
        s_accumulator[flat_idx] = sqrt(s_accumulator[flat_idx] + 1e-4);
    }

    __syncthreads();

    // // L2 norm f2 equation
    s_norm_hist[flat_idx] = s_norm_hist[flat_idx] / s_accumulator[0];

    // cooperate loading to global memory
    norm_hist[NORM_HIST_IDX(gcx, gcy, c, b)] = s_norm_hist[flat_idx];
}

#define SVM_TILE_WIDTH 128

__global__ void svm_kernel(float *X, float *W, int feature_size, float *res)
{
    extern __shared__ float s_W[];

    // Local index
    unsigned int lidx = threadIdx.x;
    // Global index
    unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    // Cooperate loading data
    s_W[lidx] = 0.0f;
    if (gidx < feature_size)
    {
        s_W[lidx] = X[gidx] * W[gidx];
    }
    __syncthreads();

    // use reduction to sum all values
    for (size_t i = 1; i < blockDim.x; i *= 2)
    {
        size_t offset = SVM_TILE_WIDTH / i / 2;
        if (lidx < offset)
        {
            s_W[lidx] += s_W[lidx + offset];
        }
        __syncthreads();
    }

    // load result to global memory
    if (lidx == 0)
    {
        res[blockIdx.x] = s_W[0];
    }
}

float launchHOG_SVM(float *img, unsigned int width, unsigned int height, float *weights, float bias, size_t num_weights, float *gx, float *gy)
{
    size_t img_chan_size = width * height * sizeof(float);
    size_t img_size = NUM_CHAN * img_chan_size;

    size_t grid_x = floor((float)width / (float)TILE_SIZE);
    size_t grid_y = floor((float)height / (float)TILE_SIZE);
    size_t hist_size = (NUM_BINS + 1) * grid_x * grid_y * sizeof(float);

    float *d_img;
    float *d_hist;

#ifdef OUT_TO_HOST
    float *d_gx;
    float *d_gy;
    cudaMalloc((void **)&d_gx, img_size);
    cudaMalloc((void **)&d_gy, img_size);
#endif

    cudaMalloc((void **)&d_img, img_size);
    cudaMalloc((void **)&d_hist, hist_size);

    cudaMemcpy(d_img, img, img_size, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 1.0f, hist_size);

    size_t tile_size = NUM_CHAN * (TILE_SIZE + 2) * (TILE_SIZE + 2) * sizeof(float);
    size_t max_grad_idx_tile_size = TILE_SIZE * TILE_SIZE * sizeof(uint8_t);
    size_t shm_size = tile_size + max_grad_idx_tile_size + hist_size;

    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, NUM_CHAN);
#ifdef OUT_TO_HOST
    histogram_cell_kernel<<<dimGrid, dimBlock, shm_size>>>(width, height, d_img, d_hist, d_gx, d_gy);
#endif
#ifndef OUT_TO_HOST
    histogram_cell_kernel<<<dimGrid, dimBlock, shm_size>>>(width, height, d_img, d_hist);
#endif

    // float *hist = (float *)malloc(hist_size);
    // cudaMemcpy(hist, d_hist, hist_size, cudaMemcpyDeviceToHost);

    size_t block_hist_num_elem = (grid_x - 1) * (grid_y - 1) * NUM_CELLS_IN_BLOCKS * NUM_BINS;
    size_t block_hist_size = block_hist_num_elem * sizeof(float);

    if (num_weights != block_hist_num_elem)
    {
        throw std::runtime_error("Number of weights is not equal to one calculated");
    }

    float *d_block_hist;
    cudaMalloc((void **)&d_block_hist, block_hist_size);

    shm_size = block_hist_size + block_hist_size;

    dim3 blockDimGrid(grid_x - 1, grid_y - 1);
    dim3 blockDimBlock(NUM_CELLS_IN_BLOCKS, NUM_BINS);
    histogram_block_kernel<<<blockDimGrid, blockDimBlock, shm_size>>>(d_hist, d_block_hist);

    // float *block_hist = (float *)malloc(block_hist_size);
    // cudaMemcpy(block_hist, d_block_hist, block_hist_size, cudaMemcpyDeviceToHost);

    size_t svm_res_num = ceil((float)block_hist_num_elem / (float)SVM_TILE_WIDTH);
    size_t svm_res_size = svm_res_num * sizeof(float);

    float *d_res;
    float *d_W;

    cudaMalloc((void **)&d_res, svm_res_size);
    cudaMalloc((void **)&d_W, block_hist_size);

    cudaMemcpy(d_W, weights, block_hist_size, cudaMemcpyHostToDevice);

    shm_size = SVM_TILE_WIDTH * sizeof(float);
    svm_kernel<<<svm_res_num, SVM_TILE_WIDTH, shm_size>>>(d_block_hist, d_W, block_hist_num_elem, d_res);

    float *res = (float *)malloc(svm_res_size);
    cudaMemcpy(res, d_res, svm_res_size, cudaMemcpyDeviceToHost);

    float predition = bias;
    for (size_t i = 0; i < svm_res_num; i++)
    {
        predition += res[i];
    }

#ifdef OUT_TO_HOST
    cudaMemcpy(gx, d_gx, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gy, d_gy, img_size, cudaMemcpyDeviceToHost);
#endif

    // free(block_hist);

    cudaFree(d_img);
    cudaFree(d_block_hist);

    cudaFree(d_res);
    cudaFree(d_W);
    // free(hist);
    free(res);

#ifdef OUT_TO_HOST
    cudaFree(d_gx);
    cudaFree(d_gy);
#endif

    return predition;
}

int main(int argc, char *argv[])
{
    // check if number of input args is correct
    if (argc != 2)
    {
        printf("Wrong number of arguments: exactly 1 arguments needed filename of input file .ppm.\n");
        return 1;
    }

    // read image sizes
    int H, W;
    getPPMSize(argv[1], &W, &H);

    // read image data
    float *inputImage = (float *)malloc(NUM_CHAN * H * W * sizeof(float));
    readPPM(argv[1], inputImage);

    float *gx = (float *)malloc(NUM_CHAN * H * W * sizeof(float));
    float *gy = (float *)malloc(NUM_CHAN * H * W * sizeof(float));

    float *weights_dynamic = (float *)malloc(3780 * sizeof(float));
    memcpy(weights_dynamic, weights, 3780 * sizeof(float));

    // run algorithm
    const int num_iter = 1000;
    float prediction = 0.0f;
    unsigned long long time = 0;
    for (size_t i = 0; i < num_iter; i++)
    {
        auto start = high_resolution_clock::now();
        prediction = launchHOG_SVM(inputImage, W, H, weights, bias, 3780, gx, gy);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        if (i > 0)
        {
            time += duration.count();
        }
    }
    std::cout << time / num_iter << std::endl;

    printf("Prediction: %f\n", prediction);

#ifdef OUT_TO_HOST
    // save outputs for the first image
    writePPM("gx.ppm", gx, W, H);
    writePPM("gy.ppm", gy, W, H);
#endif

    free(inputImage);
    return 0;
}