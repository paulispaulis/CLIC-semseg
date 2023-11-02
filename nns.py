"""
File for defining neural networks.
"""

import torch
import torch.nn.functional as f
import numpy as np
import math

import defdevice

class NN():
    def __init__(s):
        s.parameters = []
    
    def bias(s, size):
        tens = torch.zeros(size, dtype = torch.float32, requires_grad = True, device = defdevice.def_device)
        s.parameters.append(tens)
        return tens
        
    def weight(s, ins, outs):
        k = 1/math.sqrt(ins)
        data = np.random.uniform(-k, k, size = [ins, outs])
        tens = torch.tensor(data, dtype = torch.float32, requires_grad = True, device = defdevice.def_device)
        s.parameters.append(tens)
        return tens
    
    def forward(s):
        raise NotImplemented()
        
    def save(s, path):
        """
        Path shouldn't contain extension.
        """
        raise NotImplemented()
        
    def load(s, path):
        """
        Path shouldn't contain extension.
        """
        raise NotImplemented()

class Linear(NN):
    """
    Just one linear layer.
    """
    def __init__(s, ins, outs):
        super().__init__()
        s.w = s.weight(ins, outs)
        s.b = s.bias(outs)
    
    def forward(s, x):
        res = torch.matmul(x, s.w) + s.b
        return res
    
    def save(s, path):
        torch.save(s.parameters, path + '.pt')
    
    def load(s, path):
        s.parameters = torch.load(path + '.pt')
        s.w = s.parameters[0].to(defdevice.def_device)
        s.b = s.parameters[1].to(defdevice.def_device)