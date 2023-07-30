#ifndef __KALMAN_FILTER_H__
#define __KALMAN_FILTER_H__

#include "matrix.h"

typedef struct {
    // x_hat = A @ x + B @ u
    matrix_t * x_hat;
    matrix_t * A_dot_x;
    matrix_t * B_dot_u;
    // P_hat = A @ P @ A.T + Q
    matrix_t * AT;
    matrix_t * P_hat;
    matrix_t * A_dot_P;
    // y_hat = y - C @ x_hat
    matrix_t * y_hat;
    matrix_t * A_dot_P_dot_AT;
    matrix_t * C_dot_x_hat;
    // S = C @ P_hat @ C.T + R
    matrix_t * S;
    matrix_t * CT;
    matrix_t * C_dot_P_hat;
    matrix_t * C_dot_P_hat_dot_CT;
    // K = P_hat @ C.T @ S^(-1)
    matrix_t * K;
    matrix_t * inv_S;
    matrix_t * P_hat_dot_CT;
    // x = x_hat + K @ y_hat
    matrix_t * K_dot_y_hat;
    // P = (I - K @ C) @ P_hat
    matrix_t * I;
    matrix_t * K_dot_C;
    matrix_t * I_diff_K_dot_C;
} kf_int_state_t;

typedef struct {
    matrix_t * A;
    matrix_t * B;
    matrix_t * C;
    matrix_t * Q;
    matrix_t * R;
    matrix_t * P;
    matrix_t * x;
    matrix_t * y;
    matrix_t * u;
    kf_int_state_t * _int;
} kalman_filter_t;

bool kf_init(kalman_filter_t * kf);
bool kf_predict(kalman_filter_t * kf);
bool kf_destroy(kalman_filter_t * kf);

#endif // __KALMAN_FILTER_H__