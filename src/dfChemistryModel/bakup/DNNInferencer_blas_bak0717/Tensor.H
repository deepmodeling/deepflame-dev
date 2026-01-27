#pragma once
#include <cstdint>
#include <vector>
#include <numeric>
#include <functional>
#include <cmath>

template<typename DataType>
class Tensor{
private:
    std::vector<int64_t> shape_;
    int64_t element_num_;
    DataType* data_;
    bool owner_;
public:
    // std::vector<int64_t> shape_;
    // int64_t element_num_;
    // DataType* data_;
    // bool owner_;

    Tensor(std::vector<int64_t> shape):
        shape_(shape),
        element_num_(std::accumulate(shape_.begin(),shape_.end(),1,std::multiplies<int64_t>())),
        data_((DataType*)aligned_alloc(64, element_num_ * sizeof(DataType))),
        owner_(true){};

    Tensor(std::vector<int64_t> shape, DataType* data):
        shape_(shape), 
        element_num_(std::accumulate(shape_.begin(),shape_.end(),1,std::multiplies<int64_t>())),
        data_(data), 
        owner_(false){};
    ~Tensor(){
        if(owner_)  free(data_);
    }
    int64_t dim_num() const {return shape_.size();};
    int64_t element_num() const {return element_num_;};
    int64_t bytes_num() const {return element_num_ * sizeof(DataType);};
    int64_t dim(int64_t i) const {return shape_[i];};
    const DataType* data() const {return data_;};
    DataType* data() {return data_;};
    bool has_nan(){
        for(int64_t i = 0; i < element_num_; ++i){
            if(std::isnan(data_[i])){
                return true;
            }
        }
        return false;
    }
};

#ifdef _FP16_
template class Tensor<__fp16>;
#endif
template class Tensor<float>;
template class Tensor<double>;
