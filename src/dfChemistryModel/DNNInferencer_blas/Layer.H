#pragma once
#include <cstdint>
#include <cstdbool>
#include <string>
#include <vector>
#include "Tensor.H"

template<typename DataType>
class Layer{
public:
    virtual ~Layer() {};
    virtual void forward(const Tensor<DataType>& input, Tensor<DataType>& output) = 0;
    virtual void load_parameters(const std::string& dir, int64_t layer_id) = 0;
    virtual void reset_timer() = 0;
    virtual void print_timer() = 0;
};

template<typename DataType>
class Linear : public Layer<DataType>{
protected:
    int64_t in_features_;
    int64_t out_features_;
    // (in_features, out_features)
    Tensor<DataType> weights_;
    // (out_features,)
    Tensor<DataType> bias_;

    double gemm_time_;
    double bias_time_;
    double total_infer_time_;

public:
    Linear(int64_t in_features,int64_t out_features):
        in_features_(in_features),
        out_features_(out_features),
        weights_({in_features_,out_features_}),
        bias_({out_features_}){};
    virtual ~Linear(){};

    virtual void forward(const Tensor<DataType>& input, Tensor<DataType>& output);
    virtual void load_parameters(const std::string& dir, int64_t layer_id);
    virtual void reset_timer();
    virtual void print_timer();
};

template<typename DataType>
class LinearGELU : public Linear<DataType>{
protected:
    // int64_t in_features_;
    // int64_t out_features_;
    // // (in_features, out_features)
    // Tensor<DataType> weights_;
    // // (out_features,)
    // Tensor<DataType> bias_;

    // double gemm_time_;
    // double bias_time_;
    double gelu_time_;
    double bias_gelu_fusion_time_;
    // double total_infer_time_;

public:
    LinearGELU(int64_t in_features,int64_t out_features):Linear<DataType>(in_features, out_features){};
    virtual ~LinearGELU(){};

    virtual void forward(const Tensor<DataType>& input, Tensor<DataType>& output);
    virtual void reset_timer();
    virtual void print_timer();
};
