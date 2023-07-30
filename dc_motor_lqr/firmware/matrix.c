#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "matrix.h"

matrix_t * allocate_matrix(uint8_t row, uint8_t col) {
    matrix_t * M = malloc(sizeof(matrix_t));
    M->row_num = row;
    M->col_num = col;
    M->data = malloc(sizeof(double) * col * row);
    return M;
}

matrix_t * allocate_same_size_matrix(matrix_t * m) {
    matrix_t * M = malloc(sizeof(matrix_t));
    M->row_num = m->row_num;
    M->col_num = m->col_num;
    M->data = malloc(sizeof(double) * M->row_num * M->col_num);
    return M;
}

matrix_t * allocate_dot_result_matrix(matrix_t * a, matrix_t * b) {
    matrix_t * M = malloc(sizeof(matrix_t));
    M->row_num = a->row_num;
    M->col_num = b->col_num;
    M->data = malloc(sizeof(double) * M->row_num * M->col_num);
    return M;
}


void destroy_matrix(matrix_t * M) {
    free(M->data);
    free(M);
}


bool add_matrices(matrix_t * a, matrix_t * b, matrix_t * out) {
    if (a->row_num != b->row_num || a->col_num != b->col_num) {
        return false;
    }
    
    if (a->row_num != out->row_num || a->col_num != out->col_num) {
        return false;
    }

    for (size_t i = 0; i < (size_t) a->col_num*a->row_num; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return true;
}

bool diff_matrices(matrix_t * a, matrix_t * b, matrix_t * out) {
    if (a->row_num != b->row_num || a->col_num != b->col_num) {
        return false;
    }
    
    if (a->row_num != out->row_num || a->col_num != out->col_num) {
        return false;
    }

    for (size_t i = 0; i < (size_t) a->col_num*a->row_num; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return true;
}

bool elementwise_multiply(matrix_t * a, matrix_t * b, matrix_t * out) {
    if (a->row_num != b->row_num || a->col_num != b->col_num) {
        return false;
    }
    
    if (a->row_num != out->row_num || a->col_num != out->col_num) {
        return false;
    }

    for (size_t i = 0; i < (size_t) a->col_num*a->row_num; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }

    return true;
}

bool dot_product(matrix_t * a, matrix_t * b, matrix_t * out) {

    if (a->col_num != b->row_num) {
        return false;
    }
    
    if (a->row_num != out->row_num || b->col_num != out->col_num) {
        return false;
    }

    size_t cnt = 0;
    for (size_t i = 0; i < (size_t) a->row_num; i++) {
        for (size_t j = 0; j < (size_t) b->col_num; j++) {
            double acc = 0.0f;
            for (size_t k = 0; k < a->col_num; k++) {

                size_t a_adr = i*a->col_num + k;
                size_t b_adr = k*b->col_num + j;
                acc += a->data[a_adr] * b->data[b_adr];
            }

            out->data[cnt] = acc;
            cnt++;
        }
    }
    return true;
}

bool det(matrix_t * m, double * res) {
    if (m->row_num != m->col_num) {
        return false;
    }

    // Rule of Sarrus
    double counter;
    if (m->row_num == 2) {
        counter = m->data[0] * m->data[3] -
                  m->data[1] * m->data[2];
        *res = counter;
        return true;
    }

    if (m->row_num == 3) {

        counter = m->data[0] * (m->data[4]*m->data[8] - m->data[5]*m->data[7]) -
                  m->data[1] * (m->data[3]*m->data[8] - m->data[5]*m->data[6]) +
                  m->data[2] * (m->data[3]*m->data[7] - m->data[4]*m->data[6]);

        *res = counter;
        return true;
    }

    return false;
}

bool inverse_matrix(matrix_t * m, matrix_t * out) {
    if (m->row_num != m->col_num) {
        return false;
    }
    
    if (m->col_num != out->col_num || m->row_num != out->row_num) {
        return false;
    }

    if (m->row_num == 1) {
        out->data[0] = 1.0f / m->data[0];
    }

    if (m->row_num == 2) {
        double mul;
        if (!det(m, &mul)) {
            return false;
        }
        mul = 1.0f / mul;
        out->data[0] =  mul * m->data[3];
        out->data[1] = -mul * m->data[1];
        out->data[2] = -mul * m->data[2];
        out->data[3] =  mul * m->data[0];
        return true;
    }
    

    if (m->row_num == 3) {
        double mul;
        if (!det(m, &mul)) {
            return false;
        }
        mul = 1.0f / mul;
        matrix_t * minor = allocate_matrix(2, 2);
        minor->data[0] = m->data[4];
        minor->data[1] = m->data[5];
        minor->data[2] = m->data[7];
        minor->data[3] = m->data[8];
        double minor_det;
        if (!det(minor, &minor_det)) {
            return false;
        }
        out->data[0] = mul * minor_det;


        minor->data[0] = m->data[2];
        minor->data[1] = m->data[1];
        minor->data[2] = m->data[8];
        minor->data[3] = m->data[7];
        if (!det(minor, &minor_det)) {
            return false;
        }
        out->data[1] = mul * minor_det;


        minor->data[0] = m->data[1];
        minor->data[1] = m->data[2];
        minor->data[2] = m->data[4];
        minor->data[3] = m->data[5];
        if (!det(minor, &minor_det)) {
            return false;
        }
        out->data[2] = mul * minor_det;


        minor->data[0] = m->data[5];
        minor->data[1] = m->data[3];
        minor->data[2] = m->data[8];
        minor->data[3] = m->data[6];
        if (!det(minor, &minor_det)) {
            return false;
        }
        out->data[3] = mul * minor_det;


        minor->data[0] = m->data[0];
        minor->data[1] = m->data[2];
        minor->data[2] = m->data[6];
        minor->data[3] = m->data[8];
        if (!det(minor, &minor_det)) {
            return false;
        }
        out->data[4] = mul * minor_det;


        minor->data[0] = m->data[2];
        minor->data[1] = m->data[0];
        minor->data[2] = m->data[5];
        minor->data[3] = m->data[3];
        if (!det(minor, &minor_det)) {
            return false;
        }
        out->data[5] = mul * minor_det;


        minor->data[0] = m->data[3];
        minor->data[1] = m->data[4];
        minor->data[2] = m->data[6];
        minor->data[3] = m->data[7];
        if (!det(minor, &minor_det)) {
            return false;
        }
        out->data[6] = mul * minor_det;


        minor->data[0] = m->data[1];
        minor->data[1] = m->data[0];
        minor->data[2] = m->data[7];
        minor->data[3] = m->data[6];
        if (!det(minor, &minor_det)) {
            return false;
        }
        out->data[7] = mul * minor_det;


        minor->data[0] = m->data[0];
        minor->data[1] = m->data[1];
        minor->data[2] = m->data[3];
        minor->data[3] = m->data[4];
        if (!det(minor, &minor_det)) {
            return false;
        }
        out->data[8] = mul * minor_det;

        destroy_matrix(minor);

        return true;
    }

    return false;
}

bool identity(matrix_t * m) {
    if (m->row_num != m->col_num) {
        return false;
    }
    for (size_t i = 0; i < m->row_num * m->col_num; i++) {
        if (i % (m->row_num + 1) == 0) {
            m->data[i] = 1.0f;
        }
        else {
            m->data[i] = 0.0f;
        }
    }
    return true;
}

bool scalar_mul(double k, matrix_t * m, matrix_t * out) {
    if (m->row_num != out->row_num || m->col_num != out->col_num) {
        return false;
    }

    for (size_t i = 0; i < (size_t) m->col_num*m->row_num; i++) {
        out->data[i] = m->data[i] * k;
    }
    return true;
}

bool scalar_add(double k, matrix_t * m, matrix_t * out) {
    if (m->row_num != out->row_num || m->col_num != out->col_num) {
        return false;
    }

    for (size_t i = 0; i < m->col_num*m->row_num; i++) {
        out->data[i] = m->data[i] + k;
    }
    return true;
}

bool transpose(matrix_t * m, matrix_t * out) {
    if (m->col_num != out->row_num || m->row_num != out->col_num) {
        return false;
    }

    for (size_t i = 0; i < m->col_num; i++) {
        for (size_t j = 0; j < (size_t) m->row_num; j++) {
            out->data[i*m->row_num + j] = m->data[j*m->col_num + i];
        }
    }
    return true;
}

bool fill(double k, matrix_t * out) {
    for (size_t i = 0; i < (size_t) out->col_num*out->row_num; i++) {
        out->data[i] = k;
    }
    return true;
}

void print_matrix(matrix_t * m) {
    for (size_t i = 0; i < (size_t) m->row_num*m->col_num; i++) {
        printf("%7.2f ", m->data[i]);
        if ((m->col_num == 1 || i > 0) && !((i+1) % m->col_num)) {
            printf("\n");
        }
    }
}