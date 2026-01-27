import torch
import numpy as np
import time
import os
import cantera as ct

device_main = "cuda:0"
device_list = range(torch.cuda.device_count())

torch.set_printoptions(precision=10)

class NN_MLP(torch.nn.Module):
    def __init__(self, layer_info):
        super(NN_MLP, self).__init__()
        self.net = torch.nn.Sequential()
        n = len(layer_info) - 1
        for i in range(n - 1):
            self.net.add_module('linear_layer_%d' %(i), torch.nn.Linear(layer_info[i], layer_info[i + 1]))
            self.net.add_module('gelu_layer_%d' %(i), torch.nn.GELU())
            #if i <= 2:
            #    self.net.add_module('batch_norm_%d' %(i), torch.nn.BatchNorm1d(layer_info[i + 1]))
        self.net.add_module('linear_layer_%d' %(n - 1), torch.nn.Linear(layer_info[n - 1], layer_info[n]))

    def forward(self, x):
        return self.net(x)

try:
    #load variables from constant/CanteraTorchProperties
    path_r = r"./constant/CanteraTorchProperties"
    with open(path_r, "r") as f:
        data = f.read()
        i = data.index('torchModel') 
        a = data.index('"',i) 
        b = data.index('"',a+1)
        modelName = data[a+1:b]
        
        i = data.index('frozenTemperature')
        a = data.index(';', i)
        b = data.rfind(' ',i+1,a)
        frozenTemperature = float(data[b+1:a])

        i = data.index('inferenceDeltaTime')
        a = data.index(';', i)
        b = data.rfind(' ',i+1,a)
        delta_t = float(data[b+1:a])

        i = data.index('CanteraMechanismFile')
        a = data.index('"',i) 
        b = data.index('"',a+1)
        mechanismName = data[a+1:b]

        i = data.index('GPU')
        a = data.index(';', i)
        b = data.rfind(' ',i+1,a)
        switch_GPU = data[b+1:a]

    #read mechanism species number
    gas = ct.Solution(mechanismName)
    n_species = gas.n_species
    #load OpenFOAM switch
    switch_on = ["true", "True", "on", "yes", "y", "t", "any"]
    switch_off = ["false", "False", "off", "no", "n", "f", "none"]
    if switch_GPU in switch_on:
        device = torch.device(device_main)
        device_ids = device_list
    elif switch_GPU in switch_off:
        device = torch.device("cpu")
        device_ids = [0]
    else:
        print("invalid setting!")
        os._exit(0)

    lamda = 0.1
    dim = 9
    
    state_dict = torch.load(modelName,map_location='cpu')
    Xmu0 = state_dict['data_in_mean']
    Xstd0 = state_dict['data_in_std']
    Ymu0 = state_dict['data_target_mean']
    Ystd0 = state_dict['data_target_std']


    Xmu0  = torch.tensor(Xmu0).unsqueeze(0).to(device=device)
    Xstd0 = torch.tensor(Xstd0).unsqueeze(0).to(device=device)
    Ymu0  = torch.tensor(Ymu0).unsqueeze(0).to(device=device)
    Ystd0 = torch.tensor(Ystd0).unsqueeze(0).to(device=device)

    Xmu1  = Xmu0
    Xstd1 = Xstd0
    Ymu1  = Ymu0
    Ystd1 = Ystd0

    Xmu2  = Xmu0
    Xstd2 = Xstd0
    Ymu2  = Ymu0
    Ystd2 = Ystd0

    #load model  
    layers = [n_species +2, 1600, 800, 400, 1]

    model0list = []
    for i in range(n_species-1):
        model0list.append(NN_MLP(layers))
    
    for i in range(n_species-1):
        model0list[i].load_state_dict(state_dict[f'net{i}'])


    for i in range(n_species-1):
        model0list[i].eval()
        model0list[i].to(device=device)

    if len(device_ids) > 1:
        for i in range(n_species-1):
            model0list[i] = torch.nn.DataParallel(model0list[i], device_ids=device_ids)

except Exception as e:
    print(e.args)

    


def inference(vec0):
    '''
    use model to inference
    '''
    vec0 = np.abs(np.reshape(vec0, (-1, 3+n_species))) # T, P, Yi(7), Rho
    vec0[:,1] *= 101325
    mask = vec0[:,0] > frozenTemperature
    vec0_input = vec0[mask, :]
    print(f'real inference points number: {vec0_input.shape[0]}')

    try:
        with torch.no_grad():
            input0_ = torch.from_numpy(vec0_input).double().to(device=device) #cast ndarray to torch tensor


            # pre_processing
            rho0 = input0_[:, -1].unsqueeze(1)
            input0_Y = input0_[:, 2:-1].clone()
            input0_bct = input0_[:, 0:-1]
            input0_bct[:, 2:] = (input0_bct[:, 2:]**(lamda) - 1) / lamda #BCT
            input0_normalized = (input0_bct - Xmu0) / Xstd0      #DimXmu0 = 9， DimXstd0 = 9， input0_bct = 
            # input0_normalized[:, -1] = 0 #set Y_AR to 0
            input0_normalized = input0_normalized.float()
            input0_normalized = input0_normalized.to(device=device)

            #inference
            
            output0_normalized = []
            for i in range(n_species-1):
                output0_normalized.append(model0list[i](input0_normalized))
            output0_normalized = torch.cat(output0_normalized, dim=1)

            # post_processing
            output0_bct = output0_normalized * Ystd0 + Ymu0 + input0_bct[:, 2:-1]
            output0_Y = input0_Y.clone()
            output0_Y[:, :-1] = (lamda * output0_bct + 1)**(1 / lamda)
            output0_Y[:, :-1] = output0_Y[:, :-1] / torch.sum(input=output0_Y[:, :-1], dim=1, keepdim=True) * (1 - output0_Y[:, -1:])
            output0 = (output0_Y - input0_Y) * rho0 / delta_t   
            output0 = output0.cpu().numpy()


            result = np.zeros((vec0.shape[0], n_species))
            result[mask, :] = output0
            return result
    except Exception as e:
        print(e.args)

