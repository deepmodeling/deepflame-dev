#include "kernel.H"
#include "gelu_d_table.h"

template<>
double fast_exp<double>(double x){
    const double LOG2E = 1.442695040;
    const uint64_t SHIFT = static_cast<uint64_t>(1) << 52;
    x *= LOG2E;
    double xi = std::floor(x);
    double xf = x - xi;
    const double coef[2] = {-0.05288671, 0.99232129};
    double k = coef[1] * xf + coef[0] + 1.;
    uint64_t e = reinterpret_cast<const uint64_t &>(k);
    e += SHIFT * static_cast<uint64_t>(xi);
    return reinterpret_cast<double &>(e);
}

// template<>
// inline double tanh_exp<double>(double x){
//     return 1. - 2. / (exp(2. * x) + 1.);
// }

// template<>
// double tanh_exp<double>(double x){
//     if(x > 8.) return 1.;
//     if(x < -8.) return -1.;
//     return 1. - 2. / (fast_exp(2. * x) + 1.);
// }

template<>
double tanh_exp<double>(double x){
    const double max = static_cast<double>(8);
    const double zero = static_cast<double>(0);
    const double neg_one = static_cast<double>(-1.);
    const double one = static_cast<double>(1.);
    const double two = static_cast<double>(2.);
    const double sign = x < zero ? neg_one : one;
    double abs_x = std::abs(x);
    abs_x = abs_x < max ? abs_x : max;
    double abs_ret = one - two / (std::exp(two * abs_x) + one);
    return sign * abs_ret;
}

template<>
void gelu_fastexp_fusion<double>(int64_t len, double* data){
    const double const_sqrt_2_div_pi = static_cast<double>(0.7978845608028654);
    const double const_2 = static_cast<double>(0.044715);
    const double const_half = static_cast<double>(0.5);
    const double const_max = 8;
    const double const_one = 1.;
    const double const_two = 2.;
    const uint64_t const_sign_mask = 0x8000000000000000;
    const uint64_t const_abs_value_mask = 0x7FFFFFFFFFFFFFFF;
    const double const_log2e = 1.442695040;
    const double exp_coef[2] = { -0.05288671, 0.99232129};
    const uint64_t const_shift_double = 52;

#ifdef _OPENMP    
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        double x = data[i];
        double tanh_x = const_sqrt_2_div_pi * (x + const_2 * x * x * x);
        uint64_t tanh_x_int = *(uint64_t*)&tanh_x;
        uint64_t tanh_x_sign = tanh_x_int & const_sign_mask;
        uint64_t abs_tanh_x_int = tanh_x_int & const_abs_value_mask;
        double abs_tanh_x = *(double*)&abs_tanh_x_int;
        abs_tanh_x = abs_tanh_x < const_max ? abs_tanh_x : const_max;
        double exp_x = const_two * abs_tanh_x;
        exp_x *= const_log2e;
        double exp_xi = std::floor(exp_x);
        uint64_t exp_xi_int = exp_xi;
        double exp_xf = exp_x - exp_xi;
        double exp_k = exp_coef[1] * exp_xf + exp_coef[0] + const_one;
        uint64_t exp_e = *(uint64_t*)&exp_k;
        exp_e += (exp_xi_int << const_shift_double);
        double exp_ret = *(double*)&exp_e;
        double abs_ret = const_one - const_two / (exp_ret + const_one);
        uint64_t tanh_ret_int = *(uint64_t*)&abs_ret | tanh_x_sign;
        double tanh_ret = *(double*)&tanh_ret_int;
        data[i] = const_half * x * (const_one + tanh_ret);
    }
}

template<>
void gelu_lookup<double>(int64_t len, double* data){
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        double x = data[i];
        x = df_max(x, range_start);
        x = df_min(x, range_end);
        uint64_t index = (uint64_t)((x - range_start) * fit_split);
        double c0 = fast_gelu_poly_table_double[index];
        data[i] = c0;
    }
}

template<>
void bias_gelu_lookup_fusion<double>(Tensor<double>& input, const Tensor<double>& bias){
    int64_t row = input.dim(0);
    int64_t col = input.dim(1);
    int64_t ld = col;
    double* input_data = input.data();
    const double* bias_data = bias.data();
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t r = 0; r < row; ++r){
        double* input_data_row = &input_data[r * ld];
        for(int64_t c = 0; c < col; ++c){
            double x = input_data_row[c] + bias_data[c];
            x = df_max(x, range_start);
            x = df_min(x, range_end);
            uint64_t index = (uint64_t)((x - range_start) * fit_split);
            double c0 = fast_gelu_poly_table_double[index];
            input_data_row[c] = c0;
        }
    }
}

template<>
void gemm<double>(char transa, char transb, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta, double *c, int ldc){
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}
