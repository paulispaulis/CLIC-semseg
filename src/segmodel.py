"""
File for defining segmentation models.
"""

import src.defdevice as defdevice
import numpy as np
import src.tensorconversions as tensorconversions

class SegModel():
    """
    Abstract class for a model that takes an image and labels, and returns a segmentation.
    """
    def __init__(s):
        pass
    
    def forward(s, image, labels):
        """
        image - numpy array of shape [y, x, 3].
        labels - list of strings.
        
        Should return numpy array of shape [len(labels), y, x].
        """
        raise NotImplemented()
        
    def forward_multilabel(s, image, labels, aggregation = 'max'):
        """
        Segments image according to multiple labels per segmentation class.
        e.g. for RUGD6.
        
        image - numpy array of shape [y, x, 3].
        labels - list of list of strings.
        aggregation - how to smoosh together multiple label's results.
            options are - "sum", "max".
            
        Returns [len(labels), y, x].
        """
        labels_merged = sum(labels, [])
        amounts = [len(l) for l in labels]
        
        all_res = s.forward(image, labels_merged)
        
        res = np.zeros([len(labels), all_res.shape[1], all_res.shape[2]], dtype = np.float32)
        all_idx = 0
        for merged_idx, amount in enumerate(amounts):
            for idx in range(amount):
                if aggregation == 'max':
                    res[merged_idx] = np.maximum(res[merged_idx], all_res[all_idx])
                elif aggregation == 'sum':
                    res[merged_idx] = res[merged_idx] + all_res[all_idx]
                else:
                    raise ValueError('Aggregation ' + str(aggregation) + ' not recognized')
                all_idx+= 1
         
        return res

    
import torch
import torch.nn.functional as f
class CSModel(SegModel):
    """
    CLIPSchitzo model.
    """
    
    def __init__(s, image_embedder, text_embedder, transform_net):
        s.image_embedder = image_embedder
        s.text_embedder = text_embedder
        s.transform_net = transform_net
        
        s.cache_labels = ''
        s.cache = None
    
    def forward_labels(s, labels):
        if s.cache_labels != labels:
            s.cache = s.text_embedder.forward(labels)
            s.cache = torch.tensor(text_embeddings, device = defdevice.def_device, requires_grad = False)
            s.cache = s.transform_net.forward(s.cache)
        
        return s.cache
    
    def forward(s, image, labels):
        """
        image - numpy array of shape [y, x, 3]
        """
        image_embeddings = s.image_embedder.forward(image)
        text_embeddings = s.text_embedder.forward(labels)
        
        image_embeddings = torch.tensor(image_embeddings, device = defdevice.def_device, requires_grad = False)
        text_embeddings = torch.tensor(text_embeddings, device = defdevice.def_device, requires_grad = False)
        
        with torch.no_grad():
            trans_embeddings = s.transform_net.forward(text_embeddings)
            
            prods = torch.einsum('eyx,ne->nyx', image_embeddings, trans_embeddings)
            probs = torch.softmax(prods, dim = 0)
            probs = probs.unsqueeze(0)
            probs = f.interpolate(probs, size = image.shape[:2], mode = 'bilinear')
            probs = probs[0]
            probs = tensorconversions.tnp(probs)
        
        return probs