"""
File is for functions related to converting between torch and numpy tensors.
"""

def tnp(tens):
    """
    Converts tensor to a numpy array.
    Takes into account all the mess of cudas and detachments.
    """
    
    res = tens
    
    res = res.detach()
    
    res = res.to('cpu')
    
    res = res.numpy()
    
    return res