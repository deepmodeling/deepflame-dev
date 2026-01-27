#ifndef DNNThermo_blas_H
#define DNNThermo_blas_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string> 
#include <memory>
#include <cstring>
#include "Layer.H"

template<typename DataType>
class DNNThermo_blas
{
private:
    std::vector<int64_t> layers0_;
    std::vector<int64_t> layers1_;
    std::vector<Layer<DataType>*> model0_;
    std::vector<Layer<DataType>*> model1_;

    double FLOPs_per_sample0_;
    double FLOPs_per_sample1_;

    int64_t total_samples_;
    bool buffer_alloced_ = false;
    int64_t batch_size_;
    std::vector<DataType*> output_buffer0_;
    std::vector<DataType*> output_buffer1_;
public:

    int64_t input_dim0() const {return layers0_.front(); }
    int64_t output_dim0() const {return layers0_.back(); }
    int64_t input_dim1() const {return layers1_.front(); }
    int64_t output_dim1() const {return layers1_.back(); }

    int64_t batch_size() const {return batch_size_; }

    DNNThermo_blas();
    ~DNNThermo_blas();

    void load_models(const std::string dir);

    // Inference
    void Inference(
        const int64_t input_count, const DataType* input, 
        DataType* output0, DataType* output1);
};

#endif