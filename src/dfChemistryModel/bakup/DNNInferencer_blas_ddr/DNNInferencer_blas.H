#ifndef DNNInferencer_blas_H
#define DNNInferencer_blas_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string> 
#include <memory>
#include <cstring>
#include "Layer.H"


template<typename DataType>
class DNNInferencer_blas
{
private:
    std::vector<int64_t> layers_;
    std::vector<Layer<DataType>*> model0_;

    double FLOPs_per_sample_;

    int64_t total_samples_;
    bool buffer_alloced_ = false;
    int64_t batch_size_;
    std::vector<DataType*> output_buffer_;
    double theoretical_tflops_core_;
public:

    int64_t input_dim() const {return layers_.front(); }
    int64_t output_dim() const {return layers_.back(); }

    int64_t batch_size() const {return batch_size_; }

    DNNInferencer_blas();
    ~DNNInferencer_blas();

    void load_models(const std::string dir);

    // Inference
    void Inference_multiDNNs(
        const std::vector<DataType>& input0, std::vector<DataType>& output0, int64_t input_count0);

    // Inference
    void Inference_multiDNNs(
        const DataType* input0, DataType* output0, int64_t input_count0);

};

#endif