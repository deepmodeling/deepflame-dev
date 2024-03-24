#pragma once

#include <cmath>
#include <cblas.h>
#include <iostream>
#include "Tensor.H"

// template<typename T> // Todo
// static T fast_floor(T x);


#define df_max(x, y) ((x) < (y) ? (y) : (x))
#define df_min(x, y) ((x) < (y) ? (x) : (y))


template<typename T>
T tanh_exp(T x);

template<typename T>
T fast_exp(T x);

template<typename T>
void gelu_navie(int64_t len, T* data);

template<typename T>
void gelu_exp(int64_t len, T* data);

template<typename T>
void gelu_fastexp_fusion(int64_t len, T* data);

#ifdef __ARM_FEATURE_SVE
template<typename T>
void gelu_fastexp_simd(int64_t len, T* data);
#endif

template<typename T>
void gelu_lookup(int64_t len, T* data);

template<typename T>
void bias_gelu_exp_fusion(Tensor<T>& input, const Tensor<T>& bias);

template<typename T>
void bias_gelu_lookup_fusion(Tensor<T>& input, const Tensor<T>& bias);

#ifdef __cplusplus
extern "C"{
#endif

extern void sgemm_(char* transA, char* transB, int* m, int* n, int* k, float* alpha, const float* a, int* lda, const float* b, int* ldb, float* beta, float* c, int* ldc);
extern void dgemm_(char* transA, char* transB, int* m, int* n, int* k, double* alpha, const double* a, int* lda, const double* b, int* ldb, double* beta, double* c, int* ldc);

#ifdef __cplusplus
}
#endif

template<typename T>
void gemm(char transa, char transb, int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb, T beta, T *c, int ldc);

// implement ----------------------------------------------------------------------------

template<typename T>
void gelu_navie(int64_t len, T* data){
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        float x = data[i];
        data[i] = 0.5 * x * (1. + std::tanh(std::sqrt(2. / M_PI) * (x + 0.044715 * std::pow(x, 3.))));
    }
}

template<typename T>
void gelu_exp(int64_t len, T* data){
    const T const_1 = 0.7978845608028654;
    const T const_2 = 0.044715;
    const T one = 1.;
    const T half = 0.5;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        T x = data[i];
        data[i] = half * x * (one + tanh_exp<T>(const_1 * (x + const_2 * x * x * x)));
    }
}

template<typename T>
void bias_naive(Tensor<T>& input, const Tensor<T>& bias){
    int64_t row = input.dim(0);
    int64_t col = input.dim(1);
    int64_t ld = col;
    T* input_data = input.data();
    const T* bias_data = bias.data();
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t r = 0; r < row; ++r){
        for(int64_t c = 0; c < col; ++c){
            input_data[r * ld + c] += bias_data[c];
        }
    }
}

template<typename T>
void bias_gelu_exp_fusion(Tensor<T>& input, const Tensor<T>& bias){
    int64_t row = input.dim(0);
    int64_t col = input.dim(1);
    int64_t ld = col;
    const T const_1 = 0.7978845608028654;
    const T const_2 = 0.044715;
    const T one = 1.;
    const T half = 0.5;
    T* input_data = input.data();
    const T* bias_data = bias.data();
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t r = 0; r < row; ++r){
        T* input_data_row = &input_data[r * ld];
        for(int64_t c = 0; c < col; ++c){
            T x = input_data_row[c] + bias_data[c];
            input_data_row[c] = half * x * (one + tanh_exp(const_1 * (x + const_2 * x * x * x)));
        }
    }
}

#ifdef __ARM_FEATURE_SVE
template<typename T>
void gelu_fastexp_simd(int64_t len, T* data){
    return gelu_fastexp_fusion(len, data);
}
#endif


