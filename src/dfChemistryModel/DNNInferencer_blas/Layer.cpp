#include "Layer.H"
#include "kernel.H"
#include <cmath>
#include <sstream>
#include <fstream>
#include <cassert>
#include <iostream>
#include <mpi.h>
    
void read_float_data(float* data, int64_t len, std::string filepath){
    std::ifstream fin(filepath, std::ios::binary);
    if(!fin.is_open()){
        std::cerr << "open file error : " << filepath << std::endl << std::flush;
        abort();
    }
    fin.read(reinterpret_cast<char*>(data), len * sizeof(float));
    fin.close();
}

template<>
void Linear<float>::load_parameters(const std::string& dir, int64_t layer_id){
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    // if(!flag_mpi_init){
    //     std::cerr << "DNNInferencer_blas::load_models : MPI is not initialized" << std::endl;
    //     MPI_Abort(MPI_COMM_WORLD, 1);
    // }
    int mpirank;
    int mpisize;
    if(flag_mpi_init){
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    }

    if(mpirank == 0 || !flag_mpi_init){
        std::stringstream ss1,ss2;
        ss1 << dir << "/" << "linear_" << layer_id << "_weights_rowmajor_" << in_features_ << "_" << out_features_ << ".data";
        std::string weights_path = ss1.str();
        ss2 << dir << "/" << "linear_" << layer_id << "_bias_" << out_features_ << ".data";
        std::string bias_path = ss2.str();

        read_float_data(weights_.data(), weights_.element_num(), weights_path);
        read_float_data(bias_.data(), bias_.element_num(), bias_path);
    }

    if(flag_mpi_init){
        MPI_Bcast(weights_.data(), weights_.element_num(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(bias_.data(), bias_.element_num(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
}

template<>
void Linear<double>::load_parameters(const std::string& dir, int64_t layer_id){
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    // if(!flag_mpi_init){
    //     std::cerr << "DNNInferencer_blas::load_models : MPI is not initialized" << std::endl;
    //     MPI_Abort(MPI_COMM_WORLD, 1);
    // }
    int mpirank;
    int mpisize;
    if(flag_mpi_init){
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    }

    if(mpirank == 0 || !flag_mpi_init){
        std::stringstream ss1,ss2;
        ss1 << dir << "/" << "linear_" << layer_id << "_weights_rowmajor_" << in_features_ << "_" << out_features_ << ".data";
        std::string weights_path = ss1.str();
        ss2 << dir << "/" << "linear_" << layer_id << "_bias_" << out_features_ << ".data";
        std::string bias_path = ss2.str();

        float* weight_tmp = new float[weights_.element_num()];
        float* bias_tmp = new float[bias_.element_num()];

        read_float_data(weight_tmp, weights_.element_num(), weights_path);
        read_float_data(bias_tmp, bias_.element_num(), bias_path);

        double* weights_data = weights_.data();
        double* bias_data = bias_.data();
        for(int i = 0; i < weights_.element_num(); ++i)
            weights_data[i] = static_cast<double>(weight_tmp[i]);
        for(int i = 0; i < bias_.element_num(); ++i)
            bias_data[i] = static_cast<double>(bias_tmp[i]);

        delete [] weight_tmp;
        delete [] bias_tmp;
    }

    if(flag_mpi_init){
        MPI_Bcast(weights_.data(), weights_.element_num(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(bias_.data(), bias_.element_num(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

#ifdef _FP16_
template<>
void Linear<__fp16>::load_parameters(const std::string& dir, int64_t layer_id){
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(!flag_mpi_init){
        std::cerr << "DNNInferencer_blas::load_models : MPI is not initialized" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int mpirank;
    int mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    if(mpirank == 0){
        std::stringstream ss1,ss2;
        ss1 << dir << "/" << "linear_" << layer_id << "_weights_rowmajor_" << in_features_ << "_" << out_features_ << ".data";
        std::string weights_path = ss1.str();
        ss2 << dir << "/" << "linear_" << layer_id << "_bias_" << out_features_ << ".data";
        std::string bias_path = ss2.str();

        float* weight_tmp = new float[weights_.element_num()];
        float* bias_tmp = new float[bias_.element_num()];

        read_float_data(weight_tmp, weights_.element_num(), weights_path);
        read_float_data(bias_tmp, bias_.element_num(), bias_path);

        __fp16* weights_data = weights_.data();
        __fp16* bias_data = bias_.data();
        for(int i = 0; i < weights_.element_num(); ++i)
            weights_data[i] = static_cast<__fp16>(weight_tmp[i]);
        for(int i = 0; i < bias_.element_num(); ++i)
            bias_data[i] = static_cast<__fp16>(bias_tmp[i]);

        delete [] weight_tmp;
        delete [] bias_tmp;
    }
    MPI_Bcast(weights_.data(), weights_.element_num() * 2, MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(bias_.data(), bias_.element_num() * 2, MPI_BYTE, 0, MPI_COMM_WORLD);
}

#endif

template<typename DataType>
void Linear<DataType>::reset_timer(){
    gemm_time_ = 0;
    bias_time_ = 0;
    total_infer_time_ = 0;
}

template<typename DataType>
void LinearGELU<DataType>::reset_timer(){
    Linear<DataType>::gemm_time_ = 0;
    Linear<DataType>::bias_time_ = 0;
    gelu_time_ = 0;
    bias_gelu_fusion_time_ = 0;
    Linear<DataType>::total_infer_time_ = 0;
}


template<typename DataType>
void Linear<DataType>::print_timer(){
    std::cout << "total infer time : " << total_infer_time_ << std::endl;
    if(total_infer_time_ > 0.){
        std::cout << "gemm time : " << gemm_time_ << ", " << gemm_time_ * 100. / total_infer_time_ << " %" << std::endl;
        std::cout << "bias time : " << bias_time_ << ", " << bias_time_ * 100. / total_infer_time_ << " %" << std::endl;
    }
}

template<typename DataType>
void LinearGELU<DataType>::print_timer(){
    std::cout << "total infer time : " << Linear<DataType>::total_infer_time_ << std::endl;
    if(Linear<DataType>::total_infer_time_ > 0.){
        std::cout << "gemm time : " << Linear<DataType>::gemm_time_ << ", " << Linear<DataType>::gemm_time_ * 100. / Linear<DataType>::total_infer_time_ << " %" << std::endl;
        std::cout << "bias time : " << Linear<DataType>::bias_time_ << ", " << Linear<DataType>::bias_time_ * 100. / Linear<DataType>::total_infer_time_ << " %" << std::endl;
        std::cout << "gelu time : " << gelu_time_ << ", " << gelu_time_ * 100. / Linear<DataType>::total_infer_time_ << " %" << std::endl;
        std::cout << "bias gelu fusion time : " << bias_gelu_fusion_time_ << ", " << bias_gelu_fusion_time_ * 100. / Linear<DataType>::total_infer_time_ << " %" << std::endl;
    }
}

template<typename DataType>
void Linear<DataType>::forward(const Tensor<DataType>& input, Tensor<DataType>& output){
    char transA = 'N';
    char transB = 'N';
    DataType alpha = 1.;
    DataType beta = 0.;
    int m = out_features_;
    int n = input.dim(0);
    int k = in_features_;
    DataType* A = weights_.data();
    int lda = out_features_;
    DataType* B = const_cast<DataType*>(input.data());
    int ldb = input.dim(1);
    DataType*  C = output.data();
    int ldc = output.dim(1);

    double time0 = MPI_Wtime();

    gemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    double time1 = MPI_Wtime();

    bias_naive(output, bias_);
    
    double time2 = MPI_Wtime();

    gemm_time_ += time1 - time0;
    bias_time_ += time2 - time1;
    total_infer_time_ += time2 - time0;
}

template<typename DataType>
void LinearGELU<DataType>::forward(const Tensor<DataType>& input, Tensor<DataType>& output){
    char transA = 'N';
    char transB = 'N';
    DataType alpha = 1.;
    DataType beta = 0.;
    int m = Linear<DataType>::out_features_;
    int n = input.dim(0);
    int k = Linear<DataType>::in_features_;
    DataType* A = Linear<DataType>::weights_.data();
    int lda = Linear<DataType>::out_features_;
    DataType* B = const_cast<DataType*>(input.data());
    int ldb = input.dim(1);
    DataType*  C = output.data();
    int ldc = output.dim(1);

    double time0 = MPI_Wtime();

    gemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    double time1 = MPI_Wtime();

    bias_naive(output, Linear<DataType>::bias_);

    double time2 = MPI_Wtime();

    // GELU
    // gelu_navie(output.element_num(), output.data());
    // gelu_exp(output.element_num(), output.data());
    // gelu_lookup(output.element_num(), output.data());
    gelu_fastexp_fusion(output.element_num(), output.data());
    // gelu_fastexp_simd(output.element_num(), output.data());
    // bias_gelu_exp_fusion(output, bias_);
    // bias_gelu_lookup_fusion(output, bias_);
    
    double time3 = MPI_Wtime();

    Linear<DataType>::gemm_time_ += time1 - time0;
    Linear<DataType>::bias_time_ += time2 - time1;
    gelu_time_ += time3 - time2;
    
    // bias_gelu_fusion_time_ += time2 - time1;

    Linear<DataType>::total_infer_time_ += time3 - time0;
}


template class Linear<float>;
template class LinearGELU<float>;

template class Linear<double>;
template class LinearGELU<double>;

#ifdef _FP16_
template class Linear<__fp16>;
template class LinearGELU<__fp16>;
#endif