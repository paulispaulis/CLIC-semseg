"""
File intended for operations relating to messing around with images.
"""

from PIL import Image
import numpy as np

def open_image(path):
    """
    Opens an image and returns it as a numpy array.
    """
    image = Image.open(path)
    res = np.array(image)
    return res

import torch.nn.functional as f
def rescale(data, image):
    """
    Interpolates data to size of image.
    data should be tensor of size [n, y1, x1] or [y1, x1].
    image should be numpy array of size [y2, x2, 3].
    
    returns data interpolated to size [n, y2, x2] or [y2, x2].
    """
    
    if len(data.shape) == 2:
        data = data.reshape([1, 1] + list(data.shape))
        res = f.interpolate(data, size = image.shape[:2], mode = 'bilinear')
        return res[0, 0]
    
    elif len(data.shape) == 3:
        data = data.reshape([1] + list(data.shape))
        res = f.interpolate(data, size = list(image.shape[:2]), mode = 'bilinear')
        return res[0]
    
def toRGB(image):
    """
    Forces image to be of shape [y, x, 3].
    """
    #Handle grayscale images.
    if len(image.shape) == 2:
        image = np.stack([image for i in range(3)], axis = 2)

    #Handle RGBA images.
    if image.shape[2] > 3:
        image = image[:, :, :3]
    
    return image