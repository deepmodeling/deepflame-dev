#include "DNNInferencer.H"

DNNInferencer::DNNInferencer() : device_(torch::kCUDA) {}

DNNInferencer::DNNInferencer(torch::jit::script::Module torchModel)
    : torchModel_(torchModel), device_(torch::kCUDA)
{
    at::TensorOptions opts = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA);
    torchModel_.to(device_);
    Xmu_vec = torch::tensor({1933.118541482812,
                             1.2327983023706526,
                             -5.705591538151852,
                             -6.446971251373195,
                             -4.169802387800032,
                             -6.1200334699867165,
                             -4.266343396329115,
                             -2.6007437468608616,
                             -0.4049762774428252},
                            opts);
    Xstd_vec = torch::tensor({716.6568054751183,
                              0.43268544913281914,
                              2.0857655247141387,
                              2.168997234412133,
                              2.707064105162402,
                              2.2681157746245897,
                              2.221785173612795,
                              1.5510851480805254,
                              0.30283229364455927},
                             opts);
    Ymu_vec = torch::tensor({175072.98234441387,
                             125434.41067566245,
                             285397.9376620931,
                             172924.8443087139,
                             -97451.53428068386,
                             -7160.953630852251,
                             -9.791262408691773e-10},
                            opts);
    Ystd_vec = torch::tensor({179830.51132577812,
                              256152.83860126554,
                              285811.9455262339,
                              263600.5448448552,
                              98110.53711881173,
                              11752.979335965118,
                              4.0735353885293555e-09},
                             opts);
    std::cout << "load model and parameters successfully" << std::endl;
}

DNNInferencer::DNNInferencer(torch::jit::script::Module torchModel0, torch::jit::script::Module torchModel1,
                             torch::jit::script::Module torchModel2, std::string device)
    : torchModel0_(torchModel0), torchModel1_(torchModel1), torchModel2_(torchModel2), device_(device)
{
    torchModel0_.to(device_);
    torchModel1_.to(device_);
    torchModel2_.to(device_);

    // at::TensorOptions opts = at::TensorOptions().dtype(at::kDouble).device(device_);
    at::TensorOptions opts = at::TensorOptions().dtype(at::kFloat).device(device_);

    // set normalization parameters
    Xmu0_vec = torch::tensor({956.4666683951323,
                              1.2621251609602075,
                              -8.482865855078037,
                              -8.60195200775564,
                              -7.5687249938092975,
                              -8.739604352829021,
                              -3.0365348658864555,
                              -4.044646973729736,
                              -0.12868046894653598},
                             opts);
    Xstd0_vec = torch::tensor({144.56082979138094,
                               0.4316114858005481,
                               1.3421800304159297,
                               1.3271564927376922,
                               1.964747648182199,
                               1.1993472911833807,
                               1.2594695379275647,
                               1.3518816605077604,
                               0.17392016053354714},
                              opts);
    Ymu0_vec = torch::tensor({8901.112679962635,
                              27135.624769093312,
                              30141.97503208172,
                              24712.755148584696,
                              -372.9651472886253,
                              -493.34322699725413,
                              -4.31138850114707e-12},
                             opts);
    Ystd0_vec = torch::tensor({8901.112679962635,
                               27135.624769093312,
                               30141.97503208172,
                               24712.755148584696,
                               372.96514728862553,
                               493.3432269972544,
                               9.409165181242247e-11},
                              opts);
    Xmu1_vec = torch::tensor({1933.118541482812,
                              1.2327983023706526,
                              -5.705591538151852,
                              -6.446971251373195,
                              -4.169802387800032,
                              -6.1200334699867165,
                              -4.266343396329115,
                              -2.6007437468608616,
                              -0.4049762774428252},
                             opts);
    Xstd1_vec = torch::tensor({716.6568054751183,
                               0.43268544913281914,
                               2.0857655247141387,
                               2.168997234412133,
                               2.707064105162402,
                               2.2681157746245897,
                               2.221785173612795,
                               1.5510851480805254,
                               0.30283229364455927},
                              opts);
    Ymu1_vec = torch::tensor({175072.98234441387,
                              125434.41067566245,
                              285397.9376620931,
                              172924.8443087139,
                              -97451.53428068386,
                              -7160.953630852251,
                              -9.791262408691773e-10},
                             opts);
    Ystd1_vec = torch::tensor({179830.51132577812,
                               256152.83860126554,
                               285811.9455262339,
                               263600.5448448552,
                               98110.53711881173,
                               11752.979335965118,
                               4.0735353885293555e-09},
                              opts);
    Xmu2_vec = torch::tensor({2717.141719004927,
                              1.2871371577864235,
                              -5.240181052513087,
                              -4.8947914078286345,
                              -3.117070179161789,
                              -4.346362771443917,
                              -4.657258124450032,
                              -4.537442872141596,
                              -0.11656950757756744},
                             opts);
    Xstd2_vec = torch::tensor({141.48030419772115,
                               0.4281422992061657,
                               0.6561518672685264,
                               0.9820405777881894,
                               1.0442969662425572,
                               0.7554583907448359,
                               1.7144519099198097,
                               1.1299391466695952,
                               0.15743252221610685},
                              opts);
    Ymu2_vec = torch::tensor({-611.0636921032669,
                              -915.1244682112174,
                              519.5930550881994,
                              -11.949500174512165,
                              -2660.9187297995336,
                              159.56360614662788,
                              -7.136459430073843e-11},
                             opts);
    Ystd2_vec = torch::tensor({611.0636921032669,
                               915.1244682112174,
                               519.5930550881994,
                               342.3100987934528,
                               2754.8463649064784,
                               313.3717647966624,
                               2.463374792192512e-10},
                              opts);
    std::cout << "load model and parameters successfully" << std::endl;
}

DNNInferencer::~DNNInferencer() {}

// Inference
at::Tensor DNNInferencer::Inference(torch::Tensor inputs)
{
    torch::Tensor cudaInputs = inputs.to(device_);

    // generate tmpTensor
    auto Xmu_tensor = torch::unsqueeze(Xmu_vec, 0);
    auto Xstd_tensor = torch::unsqueeze(Xstd_vec, 0);
    auto Ymu_tensor = torch::unsqueeze(Ymu_vec, 0);
    auto Ystd_tensor = torch::unsqueeze(Ystd_vec, 0);

    // generate inputTensor
    torch::Tensor rhoInputs = torch::unsqueeze(cudaInputs.select(1, cudaInputs.sizes()[1] - 1), 1);
    torch::Tensor TInputs = torch::unsqueeze(cudaInputs.select(1, 0), 1);
    torch::Tensor pInputs = torch::unsqueeze(cudaInputs.select(1, 1) / 101325, 1);
    torch::Tensor YIndices = torch::linspace(2, cudaInputs.sizes()[1] - 2, cudaInputs.sizes()[1] - 3, device_).toType(torch::kLong);
    torch::Tensor YInputs = torch::index_select(cudaInputs, 1, YIndices);
    torch::Tensor YInputs_BCT = (torch::pow(YInputs, 0.1) - 1) / 0.1;

    torch::Tensor InfInputs = torch::cat({TInputs, pInputs, YInputs_BCT}, 1);
    InfInputs = (InfInputs - Xmu_tensor) / Xstd_tensor;

    InfInputs = InfInputs.toType(torch::kFloat);

    // inference and time monitor
    std::vector<torch::jit::IValue> INPUTS;
    INPUTS.push_back(InfInputs);

    at::Tensor cudaOutput = torchModel_.forward(INPUTS).toTensor();

    // generate outputTensor
    torch::Tensor deltaY = torch::index_select(cudaOutput, 1, YIndices);
    deltaY = deltaY * Ystd_tensor + Ymu_tensor;
    torch::Tensor Youtputs = torch::pow((YInputs_BCT + deltaY * 0.000001) * 0.1 + 1, 10);
    Youtputs = Youtputs / torch::sum(Youtputs, 1, 1);
    Youtputs = ((Youtputs - YInputs) * rhoInputs / 0.000001);

    return Youtputs;
}

std::vector<std::vector<double>> DNNInferencer::Inference_multiDNNs(std::vector<std::vector<double>> DNNinputs, int dimension)
{
    // generate tensor
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    torch::Tensor cudaInputs0 = torch::tensor(DNNinputs[0]).reshape({-1, dimension}).to(torch::kFloat).to(device_);
    torch::Tensor cudaInputs1 = torch::tensor(DNNinputs[1]).reshape({-1, dimension}).to(torch::kFloat).to(device_);
    torch::Tensor cudaInputs2 = torch::tensor(DNNinputs[2]).reshape({-1, dimension}).to(torch::kFloat).to(device_);

    // generate tmpTensor
    auto Xmu0_tensor = torch::unsqueeze(Xmu0_vec, 0);
    auto Xstd0_tensor = torch::unsqueeze(Xstd0_vec, 0);
    auto Ymu0_tensor = torch::unsqueeze(Ymu0_vec, 0);
    auto Ystd0_tensor = torch::unsqueeze(Ystd0_vec, 0);

    auto Xmu1_tensor = torch::unsqueeze(Xmu1_vec, 0);
    auto Xstd1_tensor = torch::unsqueeze(Xstd1_vec, 0);
    auto Ymu1_tensor = torch::unsqueeze(Ymu1_vec, 0);
    auto Ystd1_tensor = torch::unsqueeze(Ystd1_vec, 0);

    auto Xmu2_tensor = torch::unsqueeze(Xmu2_vec, 0);
    auto Xstd2_tensor = torch::unsqueeze(Xstd2_vec, 0);
    auto Ymu2_tensor = torch::unsqueeze(Ymu2_vec, 0);
    auto Ystd2_tensor = torch::unsqueeze(Ystd2_vec, 0);

    // normalization and BCT trans
    torch::Tensor rhoInputs0 = torch::unsqueeze(cudaInputs0.select(1, cudaInputs0.sizes()[1] - 1), 1);
    torch::Tensor TInputs0 = torch::unsqueeze(cudaInputs0.select(1, 0), 1);
    torch::Tensor pInputs0 = torch::unsqueeze(cudaInputs0.select(1, 1), 1);
    torch::Tensor YIndices = torch::linspace(2, cudaInputs0.sizes()[1] - 2, cudaInputs0.sizes()[1] - 3, device_).toType(torch::kLong);
    torch::Tensor YInputs0 = torch::index_select(cudaInputs0, 1, YIndices);
    torch::Tensor YInputs0_BCT = (torch::pow(YInputs0, 0.1) - 1) / 0.1;
    torch::Tensor InfInputs0 = torch::cat({TInputs0, pInputs0, YInputs0_BCT}, 1);
    InfInputs0 = (InfInputs0 - Xmu0_tensor) / Xstd0_tensor;

    torch::Tensor rhoInputs1 = torch::unsqueeze(cudaInputs1.select(1, cudaInputs1.sizes()[1] - 1), 1);
    torch::Tensor TInputs1 = torch::unsqueeze(cudaInputs1.select(1, 0), 1);
    torch::Tensor pInputs1 = torch::unsqueeze(cudaInputs1.select(1, 1), 1);
    torch::Tensor YInputs1 = torch::index_select(cudaInputs1, 1, YIndices);
    torch::Tensor YInputs1_BCT = (torch::pow(YInputs1, 0.1) - 1) / 0.1;
    torch::Tensor InfInputs1 = torch::cat({TInputs1, pInputs1, YInputs1_BCT}, 1);
    InfInputs1 = (InfInputs1 - Xmu1_tensor) / Xstd1_tensor;

    torch::Tensor rhoInputs2 = torch::unsqueeze(cudaInputs2.select(1, cudaInputs2.sizes()[1] - 1), 1);
    torch::Tensor TInputs2 = torch::unsqueeze(cudaInputs2.select(1, 0), 1);
    torch::Tensor pInputs2 = torch::unsqueeze(cudaInputs2.select(1, 1), 1);
    torch::Tensor YInputs2 = torch::index_select(cudaInputs2, 1, YIndices);
    torch::Tensor YInputs2_BCT = (torch::pow(YInputs2, 0.1) - 1) / 0.1;
    torch::Tensor InfInputs2 = torch::cat({TInputs2, pInputs2, YInputs2_BCT}, 1);
    InfInputs2 = (InfInputs2 - Xmu2_tensor) / Xstd2_tensor;

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    // std::cout << "preInf time = " << processingTime.count() << std::endl;
    time_preInf += processingTime.count();

    // inference
    std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();

    // InfInputs0 = InfInputs0.toType(torch::kFloat);
    std::vector<torch::jit::IValue> INPUTS0;
    INPUTS0.push_back(InfInputs0);
    at::Tensor cudaOutput0 = torchModel0_.forward(INPUTS0).toTensor();

    // InfInputs1 = InfInputs1.toType(torch::kFloat);
    std::vector<torch::jit::IValue> INPUTS1;
    INPUTS1.push_back(InfInputs1);
    at::Tensor cudaOutput1 = torchModel1_.forward(INPUTS1).toTensor();

    // InfInputs2 = InfInputs2.toType(torch::kFloat);
    std::vector<torch::jit::IValue> INPUTS2;
    INPUTS2.push_back(InfInputs2);
    at::Tensor cudaOutput2 = torchModel2_.forward(INPUTS2).toTensor();

    std::chrono::steady_clock::time_point stop1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime1 = std::chrono::duration_cast<std::chrono::duration<double>>(stop1 - start1);
    // std::cout << "Inf time = " << processingTime1.count() << std::endl;
    time_Inference += processingTime1.count();

    // generate outputTensor
    std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();

    std::vector<std::vector<double>> results;

    torch::Tensor deltaY0 = torch::index_select(cudaOutput0, 1, YIndices);
    deltaY0 = deltaY0 * Ystd0_tensor + Ymu0_tensor;
    torch::Tensor Youtputs0 = torch::pow((YInputs0_BCT + deltaY0 * 0.000001) * 0.1 + 1, 10);
    Youtputs0 = Youtputs0 / torch::sum(Youtputs0, 1, 1);
    Youtputs0 = ((Youtputs0 - YInputs0) * rhoInputs0 / 0.000001);

    torch::Tensor deltaY1 = torch::index_select(cudaOutput1, 1, YIndices);
    deltaY1 = deltaY1 * Ystd1_tensor + Ymu1_tensor;
    torch::Tensor Youtputs1 = torch::pow((YInputs1_BCT + deltaY1 * 0.000001) * 0.1 + 1, 10);
    Youtputs1 = Youtputs1 / torch::sum(Youtputs1, 1, 1);
    Youtputs1 = ((Youtputs1 - YInputs1) * rhoInputs1 / 0.000001);

    torch::Tensor deltaY2 = torch::index_select(cudaOutput2, 1, YIndices);
    deltaY2 = deltaY2 * Ystd2_tensor + Ymu2_tensor;
    torch::Tensor Youtputs2 = torch::pow((YInputs2_BCT + deltaY2 * 0.000001) * 0.1 + 1, 10);
    Youtputs2 = Youtputs2 / torch::sum(Youtputs2, 1, 1);
    Youtputs2 = ((Youtputs2 - YInputs2) * rhoInputs2 / 0.000001);

    std::chrono::steady_clock::time_point start3 = std::chrono::steady_clock::now();

    Youtputs0 = Youtputs0.to(torch::kDouble).to(at::kCPU);
    Youtputs1 = Youtputs1.to(torch::kDouble).to(at::kCPU);
    Youtputs2 = Youtputs2.to(torch::kDouble).to(at::kCPU);

    std::chrono::steady_clock::time_point stop3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime3 = std::chrono::duration_cast<std::chrono::duration<double>>(stop3 - start3);
    // std::cout << "hot time = " << processingTime3.count() << std::endl;
    time_hot += processingTime3.count();

    std::vector<double> RRoutputs0(Youtputs0.data_ptr<double>(), Youtputs0.data_ptr<double>() + Youtputs0.numel());
    std::vector<double> RRoutputs1(Youtputs1.data_ptr<double>(), Youtputs1.data_ptr<double>() + Youtputs1.numel());
    std::vector<double> RRoutputs2(Youtputs2.data_ptr<double>(), Youtputs2.data_ptr<double>() + Youtputs2.numel());

    results = {RRoutputs0, RRoutputs1, RRoutputs2};

    std::chrono::steady_clock::time_point stop2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime2 = std::chrono::duration_cast<std::chrono::duration<double>>(stop2 - start2);
    // std::cout << "postInf time = " << processingTime2.count() << std::endl;
    time_postInf += processingTime2.count();

    // std::cout << "preInf sum time = " << time_preInf << std::endl;
    // std::cout << "Inf sum time = " << time_Inference << std::endl;
    // std::cout << "postInf sum time = " << time_postInf << std::endl;
    // std::cout << "hot sum time = " << time_hot << std::endl;

    return results;
}
