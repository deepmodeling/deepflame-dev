#include "kernel.H"
#include <stdio.h>
#ifdef _FP16_

#include "gelu_h_table.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
// #define GELU_USE_SVE
template<>
__fp16 fast_exp<__fp16>(__fp16 x){
    const __fp16 LOG2E = 1.442695040;
    const uint16_t SHIFT = static_cast<uint16_t>(1) << 10;
    x *= LOG2E;
    __fp16 xi = std::floor(x);
    __fp16 xf = x - xi;
    const __fp16 coef[2] = {-0.05288671, 0.99232129};
    __fp16 k = coef[1] * xf + coef[0] + 1.;
    uint16_t e = reinterpret_cast<const uint16_t &>(k);
    e += SHIFT * static_cast<uint16_t>(xi);
    return reinterpret_cast<__fp16 &>(e);
}

template<>
__fp16 tanh_exp<__fp16>(__fp16 x){
    const __fp16 max = static_cast<__fp16>(5.5);
    const __fp16 zero = static_cast<__fp16>(0);
    const __fp16 neg_one = static_cast<__fp16>(-1.);
    const __fp16 one = static_cast<__fp16>(1.);
    const __fp16 two = static_cast<__fp16>(2.);
    const __fp16 sign = x < zero ? neg_one : one;
    __fp16 abs_x = std::abs(x);
    abs_x = abs_x < max ? abs_x : max;
    __fp16 abs_ret = one - two / (std::exp(two * abs_x) + one);
    return sign * abs_ret;
}

template<>
void gelu_fastexp_fusion<__fp16>(int64_t len, __fp16* data){
    const __fp16 const_sqrt_2_div_pi = static_cast<__fp16>(0.7978845608028654);
    const __fp16 const_2 = static_cast<__fp16>(0.044715);
    const __fp16 const_half = static_cast<__fp16>(0.5);
    const __fp16 const_max = 5.5;
    const __fp16 const_one = 1.;
    const __fp16 const_two = 2.;
    const uint16_t const_sign_mask = 0x8000;
    const uint16_t const_abs_value_mask = 0x7FFF;
    const __fp16 const_log2e = 1.442695040;
    const __fp16 exp_coef[2] = { -0.05288671, 0.99232129};
    const uint16_t const_shift_half = 10;

#ifdef _OPENMP    
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        __fp16 x = data[i];
        __fp16 tanh_x = const_sqrt_2_div_pi * (x + const_2 * x * x * x);
        uint16_t tanh_x_int = *(uint16_t*)&tanh_x;
        uint16_t tanh_x_sign = tanh_x_int & const_sign_mask;
        uint16_t abs_tanh_x_int = tanh_x_int & const_abs_value_mask;
        __fp16 abs_tanh_x = *(__fp16*)&abs_tanh_x_int;
        abs_tanh_x = abs_tanh_x < const_max ? abs_tanh_x : const_max;
        __fp16 exp_x = const_two * abs_tanh_x;
        exp_x *= const_log2e;
        __fp16 exp_xi = std::floor(exp_x);
        uint16_t exp_xi_int = exp_xi;
        __fp16 exp_xf = exp_x - exp_xi;
        __fp16 exp_k = exp_coef[1] * exp_xf + exp_coef[0] + const_one;
        uint16_t exp_e = *(uint16_t*)&exp_k;
        exp_e += (exp_xi_int << const_shift_half);
        __fp16 exp_ret = *(__fp16*)&exp_e;
        __fp16 abs_ret = const_one - const_two / (exp_ret + const_one);
        uint16_t tanh_ret_int = *(uint16_t*)&abs_ret | tanh_x_sign;
        __fp16 tanh_ret = *(__fp16*)&tanh_ret_int;
        data[i] = const_half * x * (const_one + tanh_ret);
    }
}

// #ifdef __ARM_FEATURE_SVE
template<>
void gelu_fastexp_simd<__fp16>(int64_t len, __fp16* data){
    const __fp16 const_sqrt_2_div_pi = 0.7978845608028654;
    const __fp16 const_2 = 0.044715;
    const __fp16 const_half = 0.5;
    const __fp16 const_max = 5.5;
    const __fp16 const_one = 1.;
    const __fp16 const_two = 2.;
    const uint16_t const_sign_mask = 0x8000;
    const uint16_t const_abs_value_mask = 0x7FFF;
    const __fp16 const_log2e = 1.442695040;
    const __fp16 const_2log2e = 2.88539008;
    const uint16_t const_shift = 10;
    const __fp16 exp_coef[2] = {-0.05288671, 0.99232129};

#ifdef _OPENMP    
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; i += svcnth() * 2){
        svbool_t p_0 = svwhilelt_b16(i, len);
        svbool_t p_1 = svwhilelt_b16(static_cast<int64_t>(i + svcnth()), len);
        svfloat16_t vgelu_x_0 = svld1(p_0, &data[i]);
        svfloat16_t vgelu_x_1 = svld1(p_1, &data[i + svcnth()]);
        svfloat16_t vgelu_x2_0 = svmul_z(p_0, vgelu_x_0, vgelu_x_0);
        svfloat16_t vgelu_x2_1 = svmul_z(p_1, vgelu_x_1, vgelu_x_1);
        svfloat16_t vgelu_x3_0 = svmul_z(p_0, vgelu_x2_0, vgelu_x_0);
        svfloat16_t vgelu_x3_1 = svmul_z(p_1, vgelu_x2_1, vgelu_x_1);
        svfloat16_t tanh_x_tmp_0_0 = svmul_z(p_0, vgelu_x3_0, const_2);
        svfloat16_t tanh_x_tmp_0_1 = svmul_z(p_1, vgelu_x3_1, const_2);
        svfloat16_t tanh_x_tmp_1_0 = svadd_z(p_0, tanh_x_tmp_0_0, vgelu_x_0);
        svfloat16_t tanh_x_tmp_1_1 = svadd_z(p_1, tanh_x_tmp_0_1, vgelu_x_1);
        svfloat16_t vtanh_x_0 = svmul_z(p_0, tanh_x_tmp_1_0, const_sqrt_2_div_pi);
        svfloat16_t vtanh_x_1 = svmul_z(p_0, tanh_x_tmp_1_1, const_sqrt_2_div_pi);
        svuint16_t vtanh_x_int_0 = *(svuint16_t*)&vtanh_x_0;
        svuint16_t vtanh_x_int_1 = *(svuint16_t*)&vtanh_x_1;
        svuint16_t vtanh_x_sign_0 = svand_z(p_0, vtanh_x_int_0, const_sign_mask);
        svuint16_t vtanh_x_sign_1 = svand_z(p_1, vtanh_x_int_1, const_sign_mask);
        svuint16_t vabs_tanh_x_int_0 = svand_z(p_0, vtanh_x_int_0, const_abs_value_mask);
        svuint16_t vabs_tanh_x_int_1 = svand_z(p_1, vtanh_x_int_1, const_abs_value_mask);
        svfloat16_t vabs_tanh_x_tmp_0 = *(svfloat16_t*)&vabs_tanh_x_int_0;
        svfloat16_t vabs_tanh_x_tmp_1 = *(svfloat16_t*)&vabs_tanh_x_int_1;
        svfloat16_t vabs_tanh_x_0 = svmin_z(p_0, vabs_tanh_x_tmp_0, const_max);
        svfloat16_t vabs_tanh_x_1 = svmin_z(p_1, vabs_tanh_x_tmp_1, const_max);
        svfloat16_t vexp_x_0 = svmul_z(p_0, vabs_tanh_x_0, const_2log2e);
        svfloat16_t vexp_x_1 = svmul_z(p_1, vabs_tanh_x_1, const_2log2e);
        svfloat16_t vexp_xi_0 = svrintm_z(p_0, vexp_x_0);
        svfloat16_t vexp_xi_1 = svrintm_z(p_1, vexp_x_1);
        svuint16_t vexp_xi_int_0 = svcvt_u16_z(p_0, vexp_xi_0);  // convert
        svuint16_t vexp_xi_int_1 = svcvt_u16_z(p_1, vexp_xi_1);  // convert
        svfloat16_t vexp_xf_0 = svsub_z(p_0, vexp_x_0, vexp_xi_0);
        svfloat16_t vexp_xf_1 = svsub_z(p_1, vexp_x_1, vexp_xi_1);
        svfloat16_t vexp_k_tmp0_0 = svmul_z(p_0, vexp_xf_0, exp_coef[1]);
        svfloat16_t vexp_k_tmp0_1 = svmul_z(p_1, vexp_xf_1, exp_coef[1]);
        svfloat16_t vexp_k_tmp1_0 = svadd_z(p_0, vexp_k_tmp0_0, exp_coef[0]);
        svfloat16_t vexp_k_tmp1_1 = svadd_z(p_1, vexp_k_tmp0_1, exp_coef[0]);
        svfloat16_t vexp_k_0 = svadd_z(p_0, vexp_k_tmp1_0, const_one);
        svfloat16_t vexp_k_1 = svadd_z(p_1, vexp_k_tmp1_1, const_one);
        svuint16_t vexp_e_tmp0_0 = *(svuint16_t*)&vexp_k_0;
        svuint16_t vexp_e_tmp0_1 = *(svuint16_t*)&vexp_k_1;
        svuint16_t vexp_e_tmp1_0 = svlsl_z(p_0, vexp_xi_int_0, const_shift); 
        svuint16_t vexp_e_tmp1_1 = svlsl_z(p_1, vexp_xi_int_1, const_shift); 
        svuint16_t vexp_e_0 = svadd_z(p_0, vexp_e_tmp0_0, vexp_e_tmp1_0);
        svuint16_t vexp_e_1 = svadd_z(p_1, vexp_e_tmp0_1, vexp_e_tmp1_1);
        svfloat16_t vexp_ret_0 = *(svfloat16_t*)&vexp_e_0;
        svfloat16_t vexp_ret_1 = *(svfloat16_t*)&vexp_e_1;
        svfloat16_t vabs_tanh_ret_tmp0_0 = svadd_z(p_0, vexp_ret_0, const_one);
        svfloat16_t vabs_tanh_ret_tmp0_1 = svadd_z(p_1, vexp_ret_1, const_one);
        svfloat16_t vabs_tanh_ret_tmp1_0 = svdivr_z(p_0, vabs_tanh_ret_tmp0_0, const_two);
        svfloat16_t vabs_tanh_ret_tmp1_1 = svdivr_z(p_1, vabs_tanh_ret_tmp0_1, const_two);
        svfloat16_t vabs_tanh_ret_0 = svsubr_z(p_0, vabs_tanh_ret_tmp1_0, const_one);
        svfloat16_t vabs_tanh_ret_1 = svsubr_z(p_1, vabs_tanh_ret_tmp1_1, const_one);
        svuint16_t vabs_tanh_ret_int_0 = *(svuint16_t*)&vabs_tanh_ret_0;
        svuint16_t vabs_tanh_ret_int_1 = *(svuint16_t*)&vabs_tanh_ret_1;
        svuint16_t vtanh_ret_int_0 = svorr_z(p_0, vabs_tanh_ret_int_0, vtanh_x_sign_0);
        svuint16_t vtanh_ret_int_1 = svorr_z(p_1, vabs_tanh_ret_int_1, vtanh_x_sign_1);
        svfloat16_t vtanh_ret_0 = *(svfloat16_t*)&vtanh_ret_int_0;
        svfloat16_t vtanh_ret_1 = *(svfloat16_t*)&vtanh_ret_int_1;
        svfloat16_t vgelu_ret_tmp_0_0 = svadd_z(p_0, vtanh_ret_0, const_one);
        svfloat16_t vgelu_ret_tmp_0_1 = svadd_z(p_1, vtanh_ret_1, const_one);
        svfloat16_t vgelu_ret_tmp_1_0 = svmul_z(p_0, vgelu_ret_tmp_0_0, vgelu_x_0);
        svfloat16_t vgelu_ret_tmp_1_1 = svmul_z(p_1, vgelu_ret_tmp_0_1, vgelu_x_1);
        svfloat16_t vgelu_ret_0 = svmul_z(p_0, vgelu_ret_tmp_1_0, const_half);
        svfloat16_t vgelu_ret_1 = svmul_z(p_1, vgelu_ret_tmp_1_1, const_half);
        svst1(p_0, &data[i], vgelu_ret_0);
        svst1(p_1, &data[i + svcnth()], vgelu_ret_1);
    }
}
// #endif


template<>
void gelu_lookup<__fp16>(int64_t len, __fp16* data){
#ifdef _OPENMP
//    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        __fp16 x = data[i];
        x = df_max(x, range_start);
        x = df_min(x, range_end);
        uint16_t index = (uint16_t)((x - range_start) * fit_split);
        __fp16 c0 = fast_gelu_poly_table_half[index];
        data[i] = c0;
    }
}

template<>
void bias_gelu_lookup_fusion<__fp16>(Tensor<__fp16>& input, const Tensor<__fp16>& bias){

    int64_t row = input.dim(0);
    int64_t col = input.dim(1);
    int64_t ld = col;
    __fp16* input_data = input.data();
    const __fp16* bias_data = bias.data();
#ifdef _OPENMP
//    #pragma omp parallel for
#endif
    for(int64_t r = 0; r < row; ++r){
        __fp16* input_data_row = &input_data[r * ld];

#ifdef GELU_USE_SVE

        const int64_t vl = svcnth();
        const int64_t vlw = svcntw();
        const int64_t step = vl * 2;

        for(int64_t c = 0; c < col; c += step){

            __fp16 x = input_data_row[c] + bias_data[c];
            x = df_max(x, range_start);
            x = df_min(x, range_end);
            uint16_t index = (uint16_t)((x - range_start) * fit_split);
            __fp16 c0 = fast_gelu_poly_table_half[index];
            input_data_row[c] = c0;

            svbool_t pg1 = svwhilelt_b16(c, col);
            svbool_t pg2 = svwhilelt_b16(c + vl, col);
            svfloat16_t input_data_row1 = svld1(pg1, input_data_row + c);
            svfloat16_t input_data_row2 = svld1(pg2, input_data_row + c + vl);
            svfloat16_t x1_tmp = svadd_z(pg1, input_data_row1, bias_data[c]);
            svfloat16_t x2_tmp = svadd_z(pg2, input_data_row2, bias_data[c]);

            svfloat16_t x1_max = svmin_z(pg1, x1_tmp, range_end);
            svfloat16_t x2_max = svmin_z(pg2, x2_tmp, range_end);
            svfloat16_t x1_v = svmax_z(pg1, x1_max, range_start);
            svfloat16_t x2_v = svmax_z(pg2, x2_max, range_start);
            svfloat16_t x1_index = svmul_z(pg1, svsub_z(pg1, x1_v, range_start), fit_split);
            svfloat16_t x2_index = svmul_z(pg2, svsub_z(pg2, x2_v, range_start), fit_split);

            svint16_t x1_index_int = svcvt_s16_z(pg1, x1_index);
            svint16_t x2_index_int = svcvt_s16_z(pg2, x2_index);
            svint32_t x1_index_int_lo = svunpklo(x1_index_int);
            svint32_t x1_index_int_hi = svunpkhi(x1_index_int);
            svint32_t x2_index_int_lo = svunpklo(x2_index_int);
            svint32_t x2_index_int_hi = svunpkhi(x2_index_int);

            svbool_t pg_1 = svwhilelt_b32(c, col);
            svbool_t pg_2 = svwhilelt_b32(c + vlw, col);
            svbool_t pg_3 = svwhilelt_b32(c + vlw * 2, col);
            svbool_t pg_4 = svwhilelt_b32(c + vlw * 3, col);

            svfloat32_t data1_lo = svld1_gather_index(pg_1, fast_gelu_poly_table_float, x1_index_int_lo);
            svfloat32_t data1_hi = svld1_gather_index(pg_2, fast_gelu_poly_table_float, x1_index_int_hi);
            svfloat32_t data2_lo = svld1_gather_index(pg_3, fast_gelu_poly_table_float, x2_index_int_lo);
            svfloat32_t data2_hi = svld1_gather_index(pg_4, fast_gelu_poly_table_float, x2_index_int_hi);
            svfloat16_t data1_lo_16 = svcvt_f16_z(pg_1, data1_lo);
            svfloat16_t data1_hi_16 = svcvt_f16_z(pg_2, data1_hi);
            svfloat16_t data2_lo_16 = svcvt_f16_z(pg_3, data2_lo);
            svfloat16_t data2_hi_16 = svcvt_f16_z(pg_4, data2_hi);

            svfloat16x2_t data1_16x2 = svcreate2(data1_lo_16, data1_hi_16);
            svfloat16x2_t data2_16x2 = svcreate2(data2_lo_16, data2_hi_16);
            svuint16_t indices = svindex_u16(0, 2);
            svfloat16_t data1 = svtbl2(data1_16x2, indices);
            svfloat16_t data2 = svtbl2(data2_16x2, indices);
            svst1_f16(pg1, input_data_row + c, data1);
            svst1_f16(pg2, input_data_row + c + vl, data2);
        }

#else
        for(int64_t c = 0; c < col; ++c){
            __fp16 x = input_data_row[c] + bias_data[c];
            x = df_max(x, range_start);
            x = df_min(x, range_end);
            uint16_t index = (uint16_t)((x - range_start) * fit_split);
            __fp16 c0 = fast_gelu_poly_table_half[index];
            input_data_row[c] = c0;
        }
#endif

    }
}

template<>
void gemm<__fp16>(char transa, char transb, int m, int n, int k, __fp16 alpha, const __fp16* a, int lda, const __fp16* b, int ldb, __fp16 beta, __fp16 *c, int ldc){
    cblas_hgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif
