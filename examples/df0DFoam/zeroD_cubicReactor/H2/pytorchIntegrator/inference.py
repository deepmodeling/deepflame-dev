from builtins import Exception, print
from calendar import prcal
import torch
import numpy as np
import math
import time
import json
import os
from easydict import EasyDict as edict
import torch.profiler
import os


torch.set_printoptions(precision=10)


class MyGELU(torch.nn.Module):
    def __init__(self):
        super(MyGELU, self).__init__()
        self.torch_PI = 3.1415926536

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / self.torch_PI) * (x + 0.044715 * torch.pow(x, 3))))


def json2Parser(json_path):
    """load json and return parser-like object"""
    with open(json_path, 'r') as f:
        args = json.load(f)
    return edict(args)

class NN_MLP(torch.nn.Module):
    def __init__(self, layer_info):
        super(NN_MLP, self).__init__()
        self.net = torch.nn.Sequential()
        n = len(layer_info) - 1
        for i in range(n - 1):
            self.net.add_module('linear_layer_%d' %(i), torch.nn.Linear(layer_info[i], layer_info[i + 1]))
            self.net.add_module('gelu_layer_%d' %(i), torch.nn.GELU())
            if i <= 2:
                self.net.add_module('batch_norm_%d' %(i), torch.nn.BatchNorm1d(layer_info[i + 1]))
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
        b = data.index('sub',a) 
        c = data.index('"',b+1)
        modelName_split1 = data[a+1:b+3]
        modelName_split2 = data[b+3:c]

        modelPath = str(modelName_split1+modelName_split2)
        model1Path = str("mechanisms/"+modelPath+"/"+modelName_split1+"1"+modelName_split2+"/checkpoint/")
        model2Path = str("mechanisms/"+modelPath+"/"+modelName_split1+"2"+modelName_split2+"/checkpoint/")
        model3Path = str("mechanisms/"+modelPath+"/"+modelName_split1+"3"+modelName_split2+"/checkpoint/")
        
        i = data.index('GPU')
        a = data.index(';', i)
        b = data.rfind(' ',i+1,a)
        switch_GPU = data[b+1:a]

    #load OpenFOAM switch
    switch_on = ["true", "True", "on", "yes", "y", "t", "any"]
    switch_off = ["false", "False", "off", "no", "n", "f", "none"]
    if switch_GPU in switch_on:
        device = torch.device("cuda")
        device_ids = range(torch.cuda.device_count())
    elif switch_GPU in switch_off:
        device = torch.device("cpu")
        device_ids = [0]
    else:
        print("invalid setting!")
        os._exit(0)



    #glbal variable will only init once when called interperter
    #load parameters from json

    lamda = 0.1
    delta_t = 1e-06
    dim = 9
    #layers = setting0.layers

    
    Xmu0 = np.load('data_in_mean.npy')
    Xstd0 = np.load('data_in_std.npy')
    Ymu0 = np.load('data_target_mean.npy')
    Ystd0 = np.load('data_target_std.npy')

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
    layers = [9, 6400, 3200, 1600, 800, 400, 6]


    model0= NN_MLP(layers) 
    model1= NN_MLP(layers) 
    model2= NN_MLP(layers) 
    if torch.cuda.is_available()==False:
        state_dict = (torch.load('Temporary_Chemical.pt',map_location='cpu'))['state_dict']
    else:
        state_dict = (torch.load('Temporary_Chemical.pt'))['state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model0.load_state_dict(new_state_dict)
    #model0.load_state_dict(state_dict)

    model1 = model0
    model2 = model0

    model0.eval()
    model0.to(device=device)
    model1.eval()
    model1.to(device=device)
    model2.eval()
    model2.to(device=device)

    if len(device_ids) > 1:
        model0 = torch.nn.DataParallel(model0, device_ids=device_ids)
        model1 = torch.nn.DataParallel(model1, device_ids=device_ids)
        model2 = torch.nn.DataParallel(model2, device_ids=device_ids)
except Exception as e:
    print(e.args)


def inference(vec0, vec1, vec2):
    '''
    use model to inference
    '''
    vec0 = np.reshape(vec0, (-1, 10)) # T, P, Yi(7), Rho
    vec1 = np.reshape(vec1, (-1, 10))
    vec2 = np.reshape(vec2, (-1, 10))
    vec0[:,1] *= 101325
    vec1[:,1] *= 101325
    vec2[:,1] *= 101325

    try:
        with torch.no_grad():
            input0_ = torch.from_numpy(vec0).double().to(device=device) #cast ndarray to torch tensor
            input1_ = torch.from_numpy(vec1).double().to(device=device) #cast ndarray to torch tensor
            input2_ = torch.from_numpy(vec2).double().to(device=device) #cast ndarray to torch tensor


            # pre_processing
            rho0 = input0_[:, -1].unsqueeze(1)
            input0_Y = input0_[:, 2:-1].clone()
            input0_bct = input0_[:, 0:-1]
            input0_bct[:, 2:] = (input0_bct[:, 2:]**(lamda) - 1) / lamda #BCT
            input0_normalized = (input0_bct - Xmu0) / Xstd0      #DimXmu0 = 9， DimXstd0 = 9， input0_bct = 
            input0_normalized = input0_normalized.float()


            rho1 = input1_[:, -1].unsqueeze(1)
            input1_Y = input1_[:, 2:-1].clone()
            input1_bct = input1_[:, 0:-1]
            input1_bct[:, 2:] = (input1_bct[:, 2:]**(lamda) - 1) / lamda #BCT
            input1_normalized = (input1_bct - Xmu1) / Xstd1
            input1_normalized = input1_normalized.float()

            rho2 = input2_[:, -1].unsqueeze(1)
            input2_Y = input2_[:, 2:-1].clone()
            input2_bct = input2_[:, 0:-1]
            input2_bct[:, 2:] = (input2_bct[:, 2:]**(lamda) - 1) / lamda #BCT
            input2_normalized = (input2_bct - Xmu2) / Xstd2
            input2_normalized = input2_normalized.float()


            
            #inference
            
            output0_normalized = model0(input0_normalized)
            output1_normalized = model1(input1_normalized)
            output2_normalized = model2(input2_normalized)


            # post_processing
            #output0_bct = (output0_normalized * Ystd0 + Ymu0) * delta_t + input0_bct
            output0_bct = output0_normalized * Ystd0 + Ymu0 + input0_bct[:, 2:-1]
            output0_Y = input0_Y.clone()
            output0_Y[:, :-1] = (lamda * output0_bct + 1)**(1 / lamda)
            output0_Y = output0_Y / torch.sum(input=output0_Y, dim=1, keepdim=True)
            output0 = (output0_Y - input0_Y) * rho0 / delta_t   
            output0 = output0.cpu().numpy()


            output1_bct = output1_normalized * Ystd1 + Ymu1 + input1_bct[:, 2:-1]
            output1_Y = input1_Y.clone()
            output1_Y[:, :-1] = (lamda * output1_bct + 1)**(1 / lamda)
            output1_Y = output1_Y / torch.sum(input=output1_Y, dim=1, keepdim=True)
            output1 = (output1_Y - input1_Y) * rho1 / delta_t   
            output1 = output1.cpu().numpy()

            output2_bct = output2_normalized * Ystd2 + Ymu2 + input2_bct[:, 2:-1]
            output2_Y = input2_Y.clone()
            output2_Y[:, :-1] = (lamda * output2_bct + 1)**(1 / lamda)
            output2_Y = output2_Y / torch.sum(input=output2_Y, dim=1, keepdim=True)
            output2 = output2_Y - output2_Y   
            output2 = output2.cpu().numpy()

            result = np.append(output0, output1, axis=0)
            result = np.append(result, output2, axis=0)
            return result
    except Exception as e:
        print(e.args)
