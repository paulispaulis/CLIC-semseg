"""
File for image<->caption datasets.
"""

class ImCapDataset():
    """
    Abstract class for image<->caption datasets.
    When trying to implement a new dataset, inherit from this class and implement the "get_next" and "reset" functions.
    """
    
    def __init__(s):
        s.sample_count = 0 #Define this value please.
    
    def get_next(s):
        """
        Function should return a touple (image, caption).
        image - numpy array of shape [y, x, 3].
        caption - string with caption.
        """
        raise NotImplemented
        
    def reset():
        """
        Functions should cause dataset to start over from the first sample.
        """
        raise NotImplemented
        
    def __str__(s):
        res = 'Image-caption dataset of class ' + str(s.__class__.__name__) + '\n'
        res+= 'Image-caption pair count: ' + str(s.sample_count) + '\n'
        
        return res
    

#This is where specific dataset implementations should start.

from datasets import load_dataset
import numpy as np
import imageops
class COCODataset(ImCapDataset):
    """
    Dataset returns image<->caption pairs.
    
    Comment - it seems like the dataset actually contains 5 captions for every image.
        So images will likely repeat.
    """
    
    def __init__(s, split = 'train', seed = 42):
        super().__init__()
        
        dataset = load_dataset("HuggingFaceM4/COCO")
        s.dataset = dataset[split].shuffle(seed)
        
        s.sample_count = s.dataset.num_rows
        s.idx = 0
        
    def get_next(s):
        image = np.array(s.dataset[s.idx]['image'])
        caption = s.dataset[s.idx]['sentences']['raw']
        
        image = imageops.toRGB(image)
        
        s.idx+= 1
        if s.idx >= s.sample_count:
            s.idx = 0
            
        return (image, caption)
    
    def reset(s):
        s.idx = 0