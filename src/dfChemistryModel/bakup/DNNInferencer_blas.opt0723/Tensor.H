#pragma once
#include <cstdint>
#include <vector>
#include <numeric>
#include <functional>
#include <cmath>
#include <hbwmalloc.h>
#include <memkind.h>
#include <assert.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>

#define HBM_ALIGNED_SIZE (1 << 21)
#define ENABLE_HBM


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
        data_(nullptr),
//        data_((DataType*)aligned_alloc(64, element_num_ * sizeof(DataType))),
        owner_(true)
    {
    
#if defined(ENABLE_HBM)
        if(element_num_ > 0) {
            int length = (element_num_ * sizeof(DataType) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
            void* mapAddress = NULL;
            mapAddress = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (mapAddress == NULL) {
                printf("mmap failed ,length = %d \n", length);
                exit(-1);
            };
            data_ = (DataType*)mapAddress;
        } else {
            data_ = nullptr;
        }
        if(owner_) printf("using mmap allocating memory\n");
#else
        data_ = (DataType*)aligned_alloc(64, element_num_ * sizeof(DataType));
#endif
    };

    Tensor(std::vector<int64_t> shape, DataType* data):
        shape_(shape), 
        element_num_(std::accumulate(shape_.begin(),shape_.end(),1,std::multiplies<int64_t>())),
        data_(data), 
        owner_(false){};
    ~Tensor(){
#if defined(ENABLE_HBM)
        if(owner_)  munmap(data_, (element_num_ * sizeof(DataType) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE);
#else
        if(owner_)  free(data_);
#endif
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
