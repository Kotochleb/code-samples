#include "kalman_filter.h"

#define CHECK_CORRECT(x) if(!(x)){return false;}

bool kf_init(kalman_filter_t * kf) {
    kf->x = allocate_matrix(kf->A->row_num, 1);
    kf->y = allocate_matrix(kf->C->row_num, 1);
    kf->u = allocate_matrix(kf->B->col_num, 1);
    kf->Q = allocate_matrix(kf->A->row_num, kf->A->row_num);
    kf->R = allocate_matrix(kf->C->row_num, kf->C->row_num);
    kf->P = allocate_matrix(kf->A->row_num, kf->A->row_num);



    kf->_int = malloc(sizeof(kf_int_state_t));
    kf->_int->x_hat = allocate_same_size_matrix(kf->x);
    kf->_int->A_dot_x = allocate_dot_result_matrix(kf->A, kf->x);
    kf->_int->B_dot_u =  allocate_dot_result_matrix(kf->B, kf->u);
    kf->_int->AT = allocate_matrix(kf->A->col_num, kf->A->row_num);
    CHECK_CORRECT(transpose(kf->A, kf->_int->AT));
    kf->_int->P_hat = allocate_same_size_matrix(kf->P);
    kf->_int->A_dot_P = allocate_dot_result_matrix(kf->A, kf->P);
    kf->_int->y_hat = allocate_same_size_matrix(kf->y);
    kf->_int->A_dot_P_dot_AT = allocate_dot_result_matrix(kf->_int->A_dot_P, kf->_int->AT);
    kf->_int->C_dot_P_hat = allocate_dot_result_matrix(kf->C, kf->_int->P_hat);
    kf->_int->C_dot_x_hat = allocate_dot_result_matrix(kf->C, kf->_int->x_hat);
    kf->_int->CT = allocate_matrix(kf->C->col_num, kf->C->row_num);
    kf->_int->C_dot_P_hat_dot_CT = allocate_dot_result_matrix(kf->_int->C_dot_P_hat, kf->_int->CT);
    kf->_int->S =  allocate_dot_result_matrix(kf->_int->C_dot_P_hat_dot_CT, kf->R);
    CHECK_CORRECT(transpose(kf->C, kf->_int->CT));
    kf->_int->P_hat_dot_CT = allocate_dot_result_matrix(kf->_int->P_hat, kf->_int->CT);
    kf->_int->inv_S = allocate_same_size_matrix(kf->_int->S);
    kf->_int->K = allocate_dot_result_matrix(kf->_int->P_hat_dot_CT, kf->_int->inv_S);
    kf->_int->K_dot_y_hat = allocate_dot_result_matrix(kf->_int->K, kf->_int->y_hat);
    kf->_int->K_dot_C = allocate_dot_result_matrix(kf->_int->K, kf->C);
    kf->_int->I = allocate_same_size_matrix(kf->_int->K_dot_C);
    CHECK_CORRECT(identity(kf->_int->I));
    kf->_int->I_diff_K_dot_C = allocate_same_size_matrix(kf->_int->I);
    return true;
}

bool kf_predict(kalman_filter_t * kf) {
    // x_hat = A @ x + B @ u
    CHECK_CORRECT(dot_product(kf->A, kf->x, kf->_int->A_dot_x));
    CHECK_CORRECT(dot_product(kf->B, kf->u, kf->_int->B_dot_u));
    CHECK_CORRECT(add_matrices(kf->_int->A_dot_x, kf->_int->B_dot_u, kf->_int->x_hat));

    // P_hat = A @ P @ A.T + Q
    CHECK_CORRECT(dot_product(kf->A, kf->P, kf->_int->A_dot_P));
    CHECK_CORRECT(dot_product(kf->_int->A_dot_P, kf->_int->AT, kf->_int->A_dot_P_dot_AT));
    CHECK_CORRECT(add_matrices(kf->_int->A_dot_P_dot_AT, kf->Q, kf->_int->P_hat));

    // y_hat = y - C @ x_hat
    CHECK_CORRECT(dot_product(kf->C, kf->_int->x_hat, kf->_int->C_dot_x_hat));
    CHECK_CORRECT(diff_matrices(kf->y, kf->_int->C_dot_x_hat, kf->_int->y_hat));

    // S = C @ P_hat @ C.T + R
    CHECK_CORRECT(dot_product(kf->C, kf->_int->P_hat, kf->_int->C_dot_P_hat));
    CHECK_CORRECT(dot_product(kf->_int->C_dot_P_hat, kf->_int->CT, kf->_int->C_dot_P_hat_dot_CT));
    CHECK_CORRECT(add_matrices(kf->_int->C_dot_P_hat_dot_CT, kf->R, kf->_int->S));

    // K = P_hat @ C.T @ S^(-1)
    CHECK_CORRECT(dot_product(kf->_int->P_hat, kf->_int->CT, kf->_int->P_hat_dot_CT));
    CHECK_CORRECT(inverse_matrix(kf->_int->S, kf->_int->inv_S));
    CHECK_CORRECT(dot_product(kf->_int->P_hat_dot_CT, kf->_int->inv_S, kf->_int->K));

    // x = x_hat + K @ y_hat
    CHECK_CORRECT(dot_product(kf->_int->K, kf->_int->y_hat, kf->_int->K_dot_y_hat));
    CHECK_CORRECT(add_matrices(kf->_int->x_hat, kf->_int->K_dot_y_hat, kf->x));

    // P = (I - K @ C) @ P_hat
    CHECK_CORRECT(dot_product(kf->_int->K, kf->C, kf->_int->K_dot_C));
    CHECK_CORRECT(diff_matrices(kf->_int->I, kf->_int->K_dot_C, kf->_int->I_diff_K_dot_C));
    CHECK_CORRECT(dot_product(kf->_int->I_diff_K_dot_C, kf->_int->P_hat, kf->P));
    return true;
}

bool kf_destroy(kalman_filter_t * kf) {
    destroy_matrix(kf->_int->x_hat);
    destroy_matrix(kf->_int->A_dot_x);
    destroy_matrix(kf->_int->B_dot_u);
    destroy_matrix(kf->_int->AT);
    destroy_matrix(kf->_int->P_hat);
    destroy_matrix(kf->_int->A_dot_P);
    destroy_matrix(kf->_int->y_hat);
    destroy_matrix(kf->_int->A_dot_P_dot_AT);
    destroy_matrix(kf->_int->C_dot_x_hat);
    destroy_matrix(kf->_int->S);
    destroy_matrix(kf->_int->CT);
    destroy_matrix(kf->_int->C_dot_P_hat);
    destroy_matrix(kf->_int->C_dot_P_hat_dot_CT);
    destroy_matrix(kf->_int->K);
    destroy_matrix(kf->_int->inv_S);
    destroy_matrix(kf->_int->P_hat_dot_CT);
    destroy_matrix(kf->_int->K_dot_y_hat);
    destroy_matrix(kf->_int->I);
    destroy_matrix(kf->_int->K_dot_C);
    destroy_matrix(kf->_int->I_diff_K_dot_C);
    free(kf->_int);

    destroy_matrix(kf->A);
    destroy_matrix(kf->B);
    destroy_matrix(kf->C);
    destroy_matrix(kf->Q);
    destroy_matrix(kf->R);
    destroy_matrix(kf->P);
    destroy_matrix(kf->x);

    return true;
}