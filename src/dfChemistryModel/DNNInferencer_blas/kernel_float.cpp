#include "kernel.H"
#include "gelu_s_table.h"

template<>
float fast_exp<float>(float x){
    const float LOG2E = 1.442695040;
    const uint32_t SHIFT = static_cast<uint32_t>(1) << 23;
    x *= LOG2E;
    float xi = std::floor(x);
    float xf = x - xi;
    const float coef[2] = {-0.05288671,0.99232129};
    float k = coef[1] * xf + coef[0] + 1.;
    uint32_t e = reinterpret_cast<const uint32_t &>(k);
    e += SHIFT * static_cast<uint32_t>(xi);
    return reinterpret_cast<float &>(e);
}

// template<>
// inline float tanh_exp<float>(float x){
//     return 1.f - 2.f / (expf(2.f * x) + 1.f);
// }

template<>
float tanh_exp<float>(float x){
    const float max = static_cast<float>(8);
    const float zero = static_cast<float>(0);
    const float neg_one = static_cast<float>(-1.);
    const float one = static_cast<float>(1.);
    const float two = static_cast<float>(2.);
    const float sign = x < zero ? neg_one : one;
    float abs_x = std::abs(x);
    abs_x = abs_x < max ? abs_x : max;
    float abs_ret = one - two / (std::exp(two * abs_x) + one);
    return sign * abs_ret;
}

template<>
void gelu_fastexp_fusion<float>(int64_t len, float* data){
    const float const_sqrt_2_div_pi = static_cast<float>(0.7978845608028654);
    const float const_2 = static_cast<float>(0.044715);
    const float const_half = static_cast<float>(0.5);
    const float const_max = 8;
    const float const_one = 1.;
    const float const_two = 2.;
    const uint32_t const_sign_mask = 0x80000000;
    const uint32_t const_abs_value_mask = 0x7FFFFFFF;
    const float const_log2e = 1.442695040;
    const float exp_coef[2] = { -0.05288671, 0.99232129};
    const uint32_t const_shift_float = 23;

#ifdef _OPENMP    
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        float x = data[i];
        float tanh_x = const_sqrt_2_div_pi * (x + const_2 * x * x * x);
        uint32_t tanh_x_int = *(uint32_t*)&tanh_x;
        uint32_t tanh_x_sign = tanh_x_int & const_sign_mask;
        uint32_t abs_tanh_x_int = tanh_x_int & const_abs_value_mask;
        float abs_tanh_x = *(float*)&abs_tanh_x_int;
        abs_tanh_x = abs_tanh_x < const_max ? abs_tanh_x : const_max;
        float exp_x = const_two * abs_tanh_x;
        exp_x *= const_log2e;
        float exp_xi = std::floor(exp_x);
        uint32_t exp_xi_int = exp_xi;
        float exp_xf = exp_x - exp_xi;
        float exp_k = exp_coef[1] * exp_xf + exp_coef[0] + const_one;
        uint32_t exp_e = *(uint32_t*)&exp_k;
        exp_e += (exp_xi_int << const_shift_float);
        float exp_ret = *(float*)&exp_e;
        float abs_ret = const_one - const_two / (exp_ret + const_one);
        uint32_t tanh_ret_int = *(uint32_t*)&abs_ret | tanh_x_sign;
        float tanh_ret = *(float*)&tanh_ret_int;
        data[i] = const_half * x * (const_one + tanh_ret);
    }
}

template<>
void gelu_lookup<float>(int64_t len, float* data){
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        float x = data[i];
        x = df_max(x, range_start);
        x = df_min(x, range_end);
        uint32_t index = (uint32_t)((x - range_start) * fit_split);
        float c0 = fast_gelu_poly_table_float[index];
        data[i] = c0;
    }
}

template<>
void bias_gelu_lookup_fusion<float>(Tensor<float>& input, const Tensor<float>& bias){
    int64_t row = input.dim(0);
    int64_t col = input.dim(1);
    int64_t ld = col;
    float* input_data = input.data();
    const float* bias_data = bias.data();
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t r = 0; r < row; ++r){
        float* input_data_row = &input_data[r * ld];
        for(int64_t c = 0; c < col; ++c){
            float x = input_data_row[c] + bias_data[c];
            x = df_max(x, range_start);
            x = df_min(x, range_end);
            uint32_t index = (uint32_t)((x - range_start) * fit_split);
            float c0 = fast_gelu_poly_table_float[index];
            input_data_row[c] = c0;
        }
    }
}


template<>
void gemm<float>(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc){
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}