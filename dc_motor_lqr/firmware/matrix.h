#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct {
    uint8_t col_num;
    uint8_t row_num;
    double *data;
} matrix_t;

matrix_t * allocate_matrix(uint8_t col, uint8_t row);
matrix_t * allocate_same_size_matrix(matrix_t * m);
matrix_t * allocate_dot_result_matrix(matrix_t * a, matrix_t * b);
void destroy_matrix(matrix_t * M);
bool add_matrices(matrix_t * a, matrix_t * b, matrix_t * out);
bool diff_matrices(matrix_t * a, matrix_t * b, matrix_t * out);
bool elementwise_multiply(matrix_t * a, matrix_t * b, matrix_t * out);
bool dot_product(matrix_t * a, matrix_t * b, matrix_t * out);
bool det(matrix_t * m, double * res);
bool inverse_matrix(matrix_t * m, matrix_t * out);
bool identity(matrix_t * out);
bool scalar_mul(double k, matrix_t * m, matrix_t * out);
bool scalar_add(double k, matrix_t * m, matrix_t * out);
bool transpose(matrix_t * m, matrix_t * out);
bool fill(double k, matrix_t * out);
void print_matrix(matrix_t * m);

#endif // __MATRIX_H__