"""
This may be a bad design choice, but I'm not particularly sure how to do this better.

File intended for globally forcing soft to run on a certain cuda device.

When definining tensors inteded for cuda in other files, it's advised to add the parameter device = defdevice.def_device.
"""

import torch

def force_device(device):
    """
    Function forces all tensors to be automatically created on the selected device.
    """
    global def_device
    def_device = device
    
    torch.cuda.set_device(def_device)

def force_cpu(device):

    global def_device
    def_device = device
=======
