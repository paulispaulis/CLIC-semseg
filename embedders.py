"""
Module for image and text embedders.

If You'd like to implement a new image embedder, inherit from ImageEmbedder, and implement the __init__ and _forward functions.
If You'd like to implement a new text embedder, inherit from TextEmbedder, and implement the __init__ and _forward functions.

You can run the test(embedder) function to run some automatic tests, for whether the embedder is acting as intended.

Current ImageEmbedders:
  M2FImageEmbedder

Current TextEmbedders:
  CLIPTextEmbedder
  
Module has sideffects because it forces all tensors to be torch.cuda.FloatTensor.
"""

import defdevice

import numpy as np
import torch
from tensorconversions import tnp
import defdevice

torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
class ImageEmbedder():
    """
    Abstract class for turning images into embedding tensors.
    In order to define an ImageEmbedder, break it into this shape.
    """
    
    def __init__(s):
        s.embedding_shape = None
        s.spacial_shape = [None, None]
        #Please define these values.
        
    def _forward(s, image):
        """
        This function should take a numpy array of the shape [x, y, 3],
        and return a embedding as a numpy array of shape [s.embedding_size, x, y].
        """
        raise NotImplemented('You forgot to implement the image embedder\'s _forward function.')
    
    def forward(s, images):
        """
        Function takes a list of numpy array of the shape [x, y, 3].
        and returns a numpy array of embedding of the shape [len(images), s.embedding_size, x,  y].
        
        Alternatively if the input is not a list, but just an image, returns [s.embedding_size, x, y].
        """
        if type(images) == list:
            res = []
            for image in images:
                res.append(s._forward(image))
            res = np.stack(res, axis = 0)
        else:
            res = s._forward(images)
        
        return res
    
    def __str__(s):
        res = 'ImageEmbedder of class ' + str(s.__class__.__name__) + '\n'
        res+= 'Embedding shape: ' + str(s.embedding_shape) + '\n'
        res+= 'Output spacial shape: ' + str(s.spacial_shape)
        
        return res
    
    
class TextEmbedder():
    """
    Abstract class for turning text into embedding tensors.
    In order to define a TextEmbedder, break it into this shape.
    """
    
    def __init__(s):
        s.embedding_shape = None
        #Please define this value
    
    def _forward(s, text):
        """
        This function should take a python string, and
        return an embedding as a numpy array of shape [s.embedding_size].
        """
        raise NotImplemented('You forgot to implement the text embedder\'s _forward function.')
    
    def forward(s, texts):
        """
        Function takes a list of python strings,
        and returns a numpy array of shape [len(text), s.embedding_size].
        
        Alternatively if text is not a list, but just a string, returns a numpy array of size [s.embedding_size].
        """
        
        if type(texts) == list:
            res = []
            for text in texts:
                res.append(s._forward(text))
            res = np.stack(res, axis = 0)
        elif type(texts) == str:
            res = s._forward(texts)
        else:
            raise TypeError('texts parameter wasn\'t string or list of string.')
            
        return res
            
    def __str__(s):
        res = 'TextEmbedder of class ' + str(s.__class__.__name__) + '\n'
        res+= 'Embedding shape: ' + str(s.embedding_shape)
        
        return res

import traceback

def test(embedder):
    """
    Just pass an instance of an ImageEmbedder or TextEmbedder class to this function, to perform tests.
    
    If the function crashes, something is wrong.
    """
    def report(exc):
        print(exc)
        print(traceback.print_exc())
    
    if isinstance(embedder, ImageEmbedder):
        print(embedder)
        
        if embedder.embedding_shape is None:
            raise Exception('Embedder doesn\'t have embedding_shape defined.')
        
        if embedder.spacial_shape == [None, None]:
            raise Exception('Embedder doesn\'t have spacial_shape defined.')
        
        try:
            res = embedder.forward([np.zeros([10, 10, 3])])[0]
        except Exception as e:
            report(e)
            raise Exception('embedder._forward() function crashes.')
        
        if res is None:
            raise Exception('embedder._forward() function doesn\'t return result.')
        
        if type(res) is not np.ndarray:
            raise Exception('Value returned by embedder._forward() isn\'t a numpy array')
        expected_shape = [embedder.embedding_shape] + embedder.spacial_shape
        
        if list(res.shape) != expected_shape:
            raise Exception('Shape of output embedder._forward() isn\'t what was expected; expected ' + str(expected_shape) + '; got ' + str(res.shape))
            
    elif isinstance(embedder, TextEmbedder):
        print(embedder)
        
        if embedder.embedding_shape is None:
            raise Exception('Embedder doesn\'t have embedding_shape defined.')
        
        try:
            res = embedder.forward(['Test String'])[0]
        except Exception as e:
            report(e)
            raise Exception('embedder._forward() function crashes.')
        
        if res is None:
            raise Exception('embedder._forward() function doesn\'t return result.')
        
        if type(res) is not np.ndarray:
            raise Exception('Value returned by embedder._forward() isn\'t a numpy array')
        expected_shape = [embedder.embedding_shape]
        
        if list(res.shape) != expected_shape:
            raise Exception('Shape of output embedder._forward() isn\'t what was expected; expected ' + str(expected_shape) + '; got ' + str(res.shape))
            
    else:
        raise ValueError('Object passed to test function doesn\'t inherit from ImageEmbedder or TextEmbedder.')
        
    print('Looks clean to me. But I\'m just a robot so what do I know.')
    

    
#Part where the specific embedder implementations should start.

class TestImageEmbedder(ImageEmbedder):
    def __init__(s):
        super().__init__()
        s.embedding_shape = 50
        s.spacial_shape = [32, 64]
        
    def _forward(s, image):
        return np.zeros([50, 32, 64])
        

class TestTextEmbedder(TextEmbedder):
    def __init__(s):
        super().__init__()
        s.embedding_shape = 50
    
    def _forward(s, text):
        return np.zeros([50])
    
    
from transformers import SamModel, SamProcessor
class SAMImageEmbedder(ImageEmbedder):
    
    def __init__(s):
        super().__init__()
        s.embedding_shape = 256
        s.spacial_shape = [64, 64]
        s.model_name = "facebook/sam-vit-huge"
        s.model = SamModel.from_pretrained(s.model_name).to(defdevice.def_device)
        s.image_processor = SamProcessor.from_pretrained(s.model_name)
        
        #This part is necessary because there's a bug in SAM's code.
        def _modified_call(
            self,
            images=None,
            input_points=None,
            input_labels=None,
            input_boxes=None,
            return_tensors = None,
            **kwargs,
        ):
            """
            This method uses [`SamImageProcessor.__call__`] method to prepare image(s) for the model. It also prepares 2D
            points and bounding boxes for the model if they are provided.
            """
            encoding_image_processor = self.image_processor(
                images,
                return_tensors=return_tensors,
                **kwargs,
            )

            # pop arguments that are not used in the foward but used nevertheless
            original_sizes = encoding_image_processor["original_sizes"]

            if hasattr(original_sizes, "numpy"):  # Checks if Torch or TF tensor
                original_sizes = original_sizes.cpu().numpy()

            input_points, input_labels, input_boxes = self._check_and_preprocess_points(
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
            )

            encoding_image_processor = self._normalize_and_convert(
                encoding_image_processor,
                original_sizes,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                return_tensors=return_tensors,
            )

            return encoding_image_processor
        
        s._modified_call = _modified_call
    
    def _forward(s, image):
        with torch.no_grad():
            inputs = s._modified_call(s.image_processor, image, return_tensors="pt").to(defdevice.def_device)
            image_embeddings = s.model.get_image_embeddings(inputs["pixel_values"])

            #normalize
            norm = torch.norm(image_embeddings[0], dim=0, p=2, keepdim=True)
            res = tnp(image_embeddings[0] / norm)
        
        return res
    
    def __str__(s):
        res = super().__str__()
        res += '\nModel name - ' + s.model_name
        
        return res
    
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
class M2FImageEmbedder(ImageEmbedder):
    def __init__(s):
        super().__init__()
        s.embedding_shape = 256
        s.spacial_shape = [96, 96]
        s.model_name = "facebook/mask2former-swin-large-coco-panoptic"
        s.image_processor = AutoImageProcessor.from_pretrained(s.model_name)
        s.model = Mask2FormerForUniversalSegmentation.from_pretrained(s.model_name).to(defdevice.def_device)
    
    def _forward(s, image):
        inputs = s.image_processor(image, return_tensors="pt").to(defdevice.def_device)
        outputs = s.model(**inputs)
        
        #normalize
        norm = torch.norm(outputs[3][0], dim=0, p=2, keepdim=True)
        res = tnp(outputs[3][0] / norm)
        
        return res
    
    def __str__(s):
        res = super().__str__()
        res += '\nModel name - ' + s.model_name
        
        return res
    
from transformers import CLIPProcessor, CLIPModel
class CLIPTextEmbedder(TextEmbedder):
    def __init__(s):
        super().__init__()
        s.embedding_shape = 512
        s.model_name = "openai/clip-vit-base-patch32"
        s.clip_model = CLIPModel.from_pretrained(s.model_name).to(defdevice.def_device)
        s.clip_processor = CLIPProcessor.from_pretrained(s.model_name)
    
    def _forward(s, text):
        inputs = s.clip_processor(text = [text], images = [np.zeros([10, 10, 3], dtype = np.float32)], return_tensors="pt", padding=True)
        outputs = s.clip_model(**inputs)
        return tnp(outputs.text_embeds[0])
    
    def __str__(s):
        res = super().__str__()
        res += '\nModel name - ' + s.model_name
        
        return res