"""
File for defining losses.
"""

import torch

def max_pressure_loss(products, value = 1.0):
    """
    Returns MSE between maximum per image and 1.
    products - tensor of shape [n, y, x].
    value - value to push max towards.
    """
    maxes = products.max(dim = 2)[0].max(dim = 1)[0]
    res = ((maxes - value) ** 2).mean()
    return res

def get_products(image_embs, text_embs):
    """
    Gets dot products between pixel embeddings and text embeddings.
    image_embs - tensor of shape [n, e, y, x].
    text_embs - tensor of shape [n, e].
    
    Returns tensor of shape [n, y, x].
    """
    prods = torch.einsum('neyx,ne->nyx', image_embs, text_embs)
    return prods

def sample_negatives(image_embs, text_embs, amount):
    """
    Samples pairs of image and text embedding tensor pairs with different indexes.
    """
    n = image_embs.shape[0]
    
    #Get indexes.
    image_idx = torch.randint(0, n, [amount])
    text_idx = (image_idx + torch.randint(1, n, [amount])).remainder(n)
    
    #Retreive embeddings by indexes.
    image_embs = image_embs[image_idx]
    text_embs = text_embs[text_idx]
    
    return (image_embs, text_embs)