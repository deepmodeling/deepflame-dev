#include "DNNThermo_blas.H"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

template<typename DataType>
DNNThermo_blas<DataType>::DNNThermo_blas() {
    char* env_tmp = getenv("DNN_BATCH_SIZE");
    if(env_tmp == NULL){
        this->batch_size_ = 16384;
    }else{
        this->batch_size_ = std::atol(env_tmp);;
    }
}

// TODO: Implement the destructor
template<typename DataType>
DNNThermo_blas<DataType>::~DNNThermo_blas() {}

template<typename DataType>
void DNNThermo_blas<DataType>::load_models(const std::string dir) {
    // mpi 
    int flag_mpi_init, mpirank, mpisize;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init){
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    }

    int32_t* count = new int32_t[2];
    char* buffer;
    std::string setting0_str, setting1_str; // two model settings

    if(mpirank == 0 || !flag_mpi_init){
        std::ifstream fin(dir + "/0/setting.yaml");
        if (!fin) {
            std::cerr << "open setting file error , setting path : " << dir + "/0/setting.yaml" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        std::ostringstream oss;
        oss << fin.rdbuf();
        fin.close();
        setting0_str = oss.str();
        count[0] = setting0_str.size();

        fin = std::ifstream(dir + "/1/setting.yaml");
        if (!fin) {
            std::cerr << "open setting file error , setting path : " << dir + "/1/setting.yaml" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        oss.str("");
        oss << fin.rdbuf();
        fin.close();
        setting1_str = oss.str();
        count[1] = setting1_str.size();

        buffer = new char[count[0] + count[1]];
        std::copy(setting0_str.begin(), setting0_str.end(), buffer);
        std::copy(setting1_str.begin(), setting1_str.end(), buffer + count[0]);
    }
    if(flag_mpi_init){
        MPI_Bcast(count, 2, MPI_INT, 0, MPI_COMM_WORLD);
        if (mpirank != 0)   buffer = new char[count[0] + count[1]];
        MPI_Bcast(buffer, count[0] + count[1], MPI_CHAR, 0, MPI_COMM_WORLD);
        if (mpirank != 0) {
            setting0_str = std::string(buffer, count[0]);
            setting1_str = std::string(buffer + count[0], count[1]);
        }
        delete[] buffer;
    }

    // init model
    YAML::Node setting0 = YAML::Load(setting0_str);
    YAML::Node layersNode0 = setting0["layers"];
    for(size_t i = 0; i < layersNode0.size(); ++i){
        layers0_.push_back(layersNode0[i].as<int64_t>());
    }

    YAML::Node setting1 = YAML::Load(setting1_str);
    YAML::Node layersNode1 = setting1["layers"];
    for(size_t i = 0; i < layersNode1.size(); ++i){
        layers1_.push_back(layersNode1[i].as<int64_t>());
    }

    // - model 0
    YAML::Node modelNode0 = setting0["model"];
    for(size_t i = 0; i < modelNode0.size(); ++i){
        std::string layerType = modelNode0[i]["layer"]["type"].as<std::string>();
        int64_t in_features = modelNode0[i]["layer"]["in_features"].as<int64_t>();
        int64_t out_features = modelNode0[i]["layer"]["out_features"].as<int64_t>();
        if(layerType == "LinearGELU"){
            model0_.push_back(new LinearGELU<DataType>(in_features, out_features));
        }else if(layerType == "Linear"){
            model0_.push_back(new Linear<DataType>(in_features, out_features));
        }else{
            assert(false);
        }
    }

    // - model 1
    YAML::Node modelNode1 = setting1["model"];
    for(size_t i = 0; i < modelNode1.size(); ++i){
        std::string layerType = modelNode1[i]["layer"]["type"].as<std::string>();
        int64_t in_features = modelNode1[i]["layer"]["in_features"].as<int64_t>();
        int64_t out_features = modelNode1[i]["layer"]["out_features"].as<int64_t>();
        if(layerType == "LinearGELU"){
            model1_.push_back(new LinearGELU<DataType>(in_features, out_features));
        }else if(layerType == "Linear"){
            model1_.push_back(new Linear<DataType>(in_features, out_features));
        }else{
            assert(false);
        }
    }

    // load parameters
    for(int i = 0; i < model0_.size(); ++i){
        model0_[i]->load_parameters(dir+"/0", i);
    }
    for(int i = 0; i < model1_.size(); ++i){
        model1_[i]->load_parameters(dir+"/1", i);
    }

    buffer_alloced_ = true;
    FLOPs_per_sample0_ = 0;
    FLOPs_per_sample1_ = 0;
    for(size_t i = 1; i < layers0_.size() - 1; ++i){
        output_buffer0_.push_back((DataType*)aligned_alloc(64, batch_size_ * layers0_[i] * sizeof(DataType)));
        FLOPs_per_sample0_ += 2.0 * layers0_[i - 1] * layers0_[i];
    }
    for(size_t i = 1; i < layers1_.size() - 1; ++i){
        output_buffer1_.push_back((DataType*)aligned_alloc(64, batch_size_ * layers1_[i] * sizeof(DataType)));
        FLOPs_per_sample1_ += 2.0 * layers1_[i - 1] * layers1_[i];
    }
}

template<typename DataType>
void DNNThermo_blas<DataType>::Inference(
    const int64_t input_count, const DataType* input, 
    DataType* output0, DataType* output1
){
    for(size_t i = 0; i < model0_.size(); ++i){
        model0_[i]->reset_timer();
    }
    for(size_t i = 0; i < model1_.size(); ++i){
        model1_[i]->reset_timer();
    }
    double dnn_infer_start = MPI_Wtime();
    // inference 
    // - NN0
    for(int64_t sample_start = 0; sample_start < input_count; sample_start += batch_size_){
        int64_t sample_end = std::min(input_count, sample_start + batch_size_);
        int64_t sample_len = sample_end - sample_start;
        std::vector<Tensor<DataType>> tensor_list;
        tensor_list.emplace_back(Tensor<DataType>({sample_len, layers0_[0]}, const_cast<DataType*>(input) + sample_start * input_dim0()));
        for(size_t i = 1; i < layers0_.size() - 1; ++i){
            tensor_list.emplace_back(Tensor<DataType>({sample_len, layers0_[i]}, output_buffer0_[i - 1]));
        }
        tensor_list.emplace_back(Tensor<DataType>({sample_len, layers0_[layers0_.size() - 1]}, output0 + sample_start * output_dim0()));

        for(size_t i = 0; i < model0_.size(); ++i){
            model0_[i]->forward(tensor_list[i], tensor_list[i+1]);
        }
    }

    // std::cout << "Done NN0 inference" << std::endl;

    // - NN1
    for(int64_t sample_start = 0; sample_start < input_count; sample_start += batch_size_){
        int64_t sample_end = std::min(input_count, sample_start + batch_size_);
        int64_t sample_len = sample_end - sample_start;
        std::vector<Tensor<DataType>> tensor_list;
        tensor_list.emplace_back(Tensor<DataType>({sample_len, layers1_[0]}, const_cast<DataType*>(input) + sample_start * input_dim1()));
        for(size_t i = 1; i < layers1_.size() - 1; ++i){
            tensor_list.emplace_back(Tensor<DataType>({sample_len, layers1_[i]}, output_buffer1_[i - 1]));
        }
        tensor_list.emplace_back(Tensor<DataType>({sample_len, layers1_[layers1_.size() - 1]}, output1 + sample_start * output_dim1()));
        for(size_t i = 0; i < model1_.size(); ++i){
            model1_[i]->forward(tensor_list[i], tensor_list[i+1]);
        }
    }
    double dnn_infer_end = MPI_Wtime();
    double dnn_infer_time = dnn_infer_end - dnn_infer_start;
    double FLOPs = input_count * FLOPs_per_sample0_ + input_count * FLOPs_per_sample1_;
    int num_threads = omp_get_max_threads();
    double theoretical_peak = 3.3792 / 48. * num_threads;
    if(sizeof(DataType) == sizeof(double)){
    }else if(sizeof(DataType) == sizeof(float)){
        theoretical_peak *= 2.;
    }
#ifdef _FP16_
    else if(sizeof(DataType) == sizeof(__fp16)){
        theoretical_peak *= 4.;
    }
#endif
    else{
        assert(false);
    }

    double FLOPS = FLOPs / dnn_infer_time;
    double TFLOPS = FLOPS * 1e-12;
    double peak = TFLOPS * 100. / theoretical_peak;

    int mpirank;
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);

    if(flag_mpi_init) MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

    if(mpirank == 0 || !flag_mpi_init){
        std::cout << "Inference Performance ---------------" << std::endl;
        std::cout << "samples : " << input_count << std::endl;
        std::cout << "batch size : " << batch_size_ << std::endl;
        std::cout << "Time : " << dnn_infer_time << std::endl;
        std::cout << "FLOPS : " << FLOPs << std::endl;
        std::cout << "TFLOPS : " << TFLOPS << std::endl;
        std::cout << "Theoretical peak : " << theoretical_peak << std::endl;
        std::cout << "Peak : " << peak << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        for(size_t i = 0; i < model0_.size(); ++i){
            model0_[i]->print_timer();
            std::cout << "-------------------------------------" << std::endl;
        }
        for(size_t i = 0; i < model1_.size(); ++i){
            model1_[i]->print_timer();
            std::cout << "-------------------------------------" << std::endl;
        }
    }
}

template class DNNThermo_blas<float>;
template class DNNThermo_blas<double>;
#ifdef _FP16_
template class DNNThermo_blas<__fp16>;
#endif