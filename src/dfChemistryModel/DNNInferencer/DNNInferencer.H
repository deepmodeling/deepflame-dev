#ifndef DNNInferencer_H
#define DNNInferencer_H

#include <torch/script.h>
#include <iostream>
#include <string> 
#include <memory>

class DNNInferencer
{
private:
    torch::jit::script::Module torchModel_, torchModel0_, torchModel1_, torchModel2_;
    torch::Device device_;
    torch::Tensor Xmu_vec, Xstd_vec, Ymu_vec, Ystd_vec,
                  Xmu0_vec, Xstd0_vec, Ymu0_vec, Ystd0_vec,
                  Xmu1_vec, Xstd1_vec, Ymu1_vec, Ystd1_vec,
                  Xmu2_vec, Xstd2_vec, Ymu2_vec, Ystd2_vec;
    double time_preInf = 0;
    double time_Inference = 0;
    double time_postInf = 0;
    double time_hot = 0;

public:
    DNNInferencer();
    DNNInferencer(torch::jit::script::Module torchModel);
    DNNInferencer(torch::jit::script::Module torchModel0, torch::jit::script::Module torchModel1, torch::jit::script::Module torchModel2, std::string device);
    ~DNNInferencer();

    // Inference
    at::Tensor Inference(torch::Tensor inputs);
    std::vector<std::vector<double>> Inference_multiDNNs(std::vector<std::vector<double>> DNNinputs, int dimension);
};

#endif