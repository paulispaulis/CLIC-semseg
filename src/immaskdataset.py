"""
File is for image<->segmentation mask datasets.
If You'd wish to add a new dataset, please inherit from this class and implement the __init__, get_next and reset functions.
Run test(<dataset instance>) to do a quick test of the class.
"""

class ImMaskDataset():
    """
    Abstract class for image<->segmask dataset.
    """
    
    def __init__(s):
        s.sample_count = 0 #Define these values please.
        s.class_count = 0
        s.labels = []
        
    def get_next(s):
        """
        Function should return a tuple (image, masks).
        image - np array of shape [x, y, 3]. (numbers in range [0..256])
        masks - np array of shape [s.class_count, x, y].
        """
        raise NotImplemented
    
    def reset(s):
        """
        Running function should make dataset start over from the first datapoint.
        """
        raise NotImplemented
    
    def __str__(s):
        res = 'ImMaskDataset of Class ' + str(s.__class__.__name__) + '\n'
        res+= 'Image segmentation pair count ' + str(s.sample_count) + '\n'
        res+= 'Class count ' + str(s.class_count) + '\n'
        
        return res
    
    
def test(ds):
    """
    Function tests instances of ImMaskDataset.
    ds - an instance of your dataset.
    Throws errors if something went wrong.
    """
    
    raise NotImplemented
    
    
#This is where specific implementations should start.

def zero_pad(number, digits = 8):
    """
    Pads positive natural number with zeros in the front to a string length of 'digits'.
    Necessary for ADEDataset.
    """
    res = ''
    
    l = 0
    tmp = number
    while tmp > 0:
        tmp//= 10
        l+= 1
        
    for i in range(digits - l):
        res+= '0'
    
    res+= str(number)
    
    return res

import numpy as np
class ADEDataset(ImMaskDataset):
    """
    ADE20k dataset.
    
    Comments: 
        Depending on the problem, handling of the "no-object" maksk (s.labels[0]) might be undersirable for your application.
    """
    
    def __init__(s, split = 'train', seed = 42, path = './ADEChallengeData2016/'):
        s.path = path
        
        if split == 'train':
            s.sample_count = 20210
            
            #This is stuff relating to image paths.
            s.folder = 'training/'
            s.image_infix = 'train'

        elif split == 'validation':
            s.sample_count = 2000
            
            s.folder = 'validation/'
            s.image_infix = 'val'
            
        else:
            raise ValueError('ADE20k dataset split ' + split + 'not recognized (expected \'train\' or \'validation\')')

        #Shuffling indexes.
        s.permutation = np.random.permutation(s.sample_count) + 1
        s.idx = 0
        
        s._load_labels()
        
    def _load_labels(s):
        s.class_count = 151
        import csv
        file_path = s.path + 'objectInfo150.txt'
        with open(file_path, 'r') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            res = []
            #These are my best guesses for what objectInfo150.txt contains:
            s.ratio = [-1] #Some kind of proportion of how much this tag is represented in dataset.
            s.train_instances = [-1] #How much tag appears in train dataset.
            s.val_instances = [-1] #-||- validation dataset
            s.labels = [''] #Actual text of the object category.
            first_row = True
            for row in reader:
                #Special handling for first row, because it contains column names.
                if first_row:
                    first_row = False
                    continue
                    
                s.ratio.append(float(row[1]))
                s.train_instances.append(int(row[2]))
                s.val_instances.append(int(row[3]))
                s.labels.append(row[4])
        
    def get_next(s):
        import imageops
        
        #Get image.
        image_number = s.permutation[s.idx]
        image = imageops.open_image(s.path + 'images/' + s.folder + 'ADE_' + s.image_infix + '_' + zero_pad(image_number) + '.jpg')
        image = imageops.toRGB(image)
        
        #Get masks
        annotation_image = imageops.open_image(s.path + 'annotations/' + s.folder + 'ADE_' + s.image_infix + '_' + zero_pad(image_number) + '.png')
        shape = annotation_image.shape
        masks = np.zeros((s.class_count, shape[0], shape[1]), dtype=np.uint8)
        for i in range(s.class_count):
            masks[i] = (annotation_image == i).astype(np.uint8)

        #Index updating.
        s.idx+= 1
        if s.idx >= s.sample_count:
            s.idx = 0
        
        return (image, masks)
    
    def reset(s):
        s.idx = 0
        
        
import os
import imageops
import numpy as np
class RUGDDataset(ImMaskDataset):
    def __init__(self, path = './RUGD/'):
        
        self.sample_count = 7436
        
        RUGD_pics_path = path + 'RUGD_frames-with-annotations/'
        RUGD_annotations_path = path + 'RUGD_annotations/'
        
        self.reset()
        self.annotation_info(RUGD_annotations_path + 'RUGD_annotation-colormap.txt')
        
        self.image_paths = []
        self.annotation_paths = []
        
        for directory, subdirlist, filelist in os.walk(RUGD_pics_path):
            
            subdir = directory[len(RUGD_pics_path):]
            
            for f in filelist:
                
                image_path = os.path.join(directory, f)
                annotation_path = os.path.join(RUGD_annotations_path, subdir, f)
                
                self.image_paths.append(image_path)
                self.annotation_paths.append(annotation_path)
                
    def annotation_info(self, filepath):
        
        self.labels = []
        self.pixel_values = []

        with open(filepath, 'r') as file:
            for line in file:
                data = line.strip().split()
                self.labels.append(data[1])
                self.pixel_values.append([int(data[2]), int(data[3]), int(data[4])])
                
        self.class_count = len(self.labels)
        
    def reset(self):
        
        self.idx = 0
        
    def fill_mask_with_color(self, annotation, mask, pixel_color):
        # Get the shape of the annotation
        height, width, _ = annotation.shape
    
        mask+= np.all((annotation == pixel_color), axis = 2).astype(np.uint8)
    
        return mask
        
        
    def get_next(self):
        
        image_path = self.image_paths[self.idx]
        annotation_path = self.annotation_paths[self.idx]
        
        image = imageops.open_image(image_path)
        annotation = imageops.open_image(annotation_path)
        
        shape = annotation.shape
        masks = np.zeros((self.class_count, shape[0], shape[1]), dtype=np.uint8)
        
        for i, colour in enumerate(self.pixel_values):
            masks[i] = self.fill_mask_with_color(annotation, masks[i], colour)
            
        self.idx += 1
        
        return (image, masks)
    
    
class RUGD6Dataset(ImMaskDataset):
    def __init__(self, path = './RUGD/'):
        
        self.rough_labels = ['background', 'smooth terrain', 'rough terrain', 'bumpy terrain', 'forbidden terrain', 'obstacle']
        self.merges = [0, 2, 2, 2, 5, 5, 4, 0, 5, 5, 1, 2, 5, 2, 3, 5, 5, 5, 5, 5, 0, 3, 5, 1, 5]
        
        self.sample_count = 7436
        
        RUGD_pics_path = path + 'RUGD_frames-with-annotations/'
        RUGD_annotations_path = path + 'RUGD_annotations/'
        
        self.reset()
        self.annotation_info(RUGD_annotations_path + 'RUGD_annotation-colormap.txt')
        
        self.image_paths = []
        self.annotation_paths = []
        
        for directory, subdirlist, filelist in os.walk(RUGD_pics_path):
            
            subdir = directory[len(RUGD_pics_path):]
            
            for f in filelist:
                
                image_path = os.path.join(directory, f)
                annotation_path = os.path.join(RUGD_annotations_path, subdir, f)
                
                self.image_paths.append(image_path)
                self.annotation_paths.append(annotation_path)
                
    def annotation_info(self, filepath):
        
        self.labels = []
        self.pixel_values = []

        with open(filepath, 'r') as file:
            for line in file:
                data = line.strip().split()
                self.labels.append(data[1])
                self.pixel_values.append([int(data[2]), int(data[3]), int(data[4])])
                
        #Merge labels
        labels = [[] for i in range(6)]
        for idx in range(len(self.labels)):
            labels[self.merges[idx]].append(self.labels[idx])
        self.labels = labels
                
        self.class_count = len(self.labels)
        
    def reset(self):
        
        self.idx = 0
        
    def fill_mask_with_color(self, annotation, mask, pixel_color):
        # Get the shape of the annotation
        height, width, _ = annotation.shape
    
        mask+= np.all((annotation == pixel_color), axis = 2).astype(np.uint8)
    
        return mask
        
        
    def get_next(self):
        
        image_path = self.image_paths[self.idx]
        annotation_path = self.annotation_paths[self.idx]
        
        image = imageops.open_image(image_path)
        annotation = imageops.open_image(annotation_path)
        
        shape = annotation.shape
        masks = np.zeros((25, shape[0], shape[1]), dtype=np.uint8)
        
        for i, colour in enumerate(self.pixel_values):
            masks[i] = self.fill_mask_with_color(annotation, masks[i], colour)
        
        def num_or(x, y):
            return 1 - (1 - x) * (1 - y)
        
        #Merge Masks.
        res_mask = np.zeros([6, shape[0], shape[1]], dtype = np.float32)
        for i in range(25):
            midx = self.merges[i]
            res_mask[midx] = num_or(res_mask[midx], masks[i])
        
            
        self.idx += 1
        
        return (image, res_mask)

class CursedRUGDDataset(ImMaskDataset):
    def __init__(self, file_list, path = './RUGD/'):
        
        self.sample_count = len(file_list)
        
        RUGD_pics_path = path + 'RUGD_frames-with-annotations/'
        RUGD_annotations_path = path + 'RUGD_annotations/'
        
        self.reset()
        self.annotation_info(RUGD_annotations_path + 'RUGD_annotation-colormap.txt')
        
        self.image_paths = []
        self.annotation_paths = []

        self.image_paths = [RUGD_pics_path + f + '.png'for f in file_list]
        self.annotation_paths = [RUGD_annotations_path + f + '.png' for f in file_list]

        """
        for directory, subdirlist, filelist in os.walk(RUGD_pics_path):
            
            subdir = directory[len(RUGD_pics_path):]
            
            for f in filelist:
                
                image_path = os.path.join(directory, f)
                annotation_path = os.path.join(RUGD_annotations_path, subdir, f)
                
                self.image_paths.append(image_path)
                self.annotation_paths.append(annotation_path)
        """
                
    def annotation_info(self, filepath):
        
        self.labels = []
        self.pixel_values = []

        with open(filepath, 'r') as file:
            for line in file:
                data = line.strip().split()
                self.labels.append(data[1])
                self.pixel_values.append([int(data[2]), int(data[3]), int(data[4])])
                
        self.class_count = len(self.labels)
        
    def reset(self):
        
        self.idx = 0
        
    def fill_mask_with_color(self, annotation, mask, pixel_color):
        # Get the shape of the annotation
        height, width, _ = annotation.shape
    
        mask+= np.all((annotation == pixel_color), axis = 2).astype(np.uint8)
    
        return mask
        
        
    def get_next(self):
        
        image_path = self.image_paths[self.idx]
        annotation_path = self.annotation_paths[self.idx]
        
        image = imageops.open_image(image_path)
        annotation = imageops.open_image(annotation_path)
        
        shape = annotation.shape
        masks = np.zeros((self.class_count, shape[0], shape[1]), dtype=np.uint8)
        
        for i, colour in enumerate(self.pixel_values):
            masks[i] = self.fill_mask_with_color(annotation, masks[i], colour)
            
        self.idx += 1
        
        return (image, masks)
    
class CursedRUGD6Dataset(ImMaskDataset):
    def __init__(self, file_list, path = './RUGD/'):
        
        self.rough_labels = ['background', 'smooth terrain', 'rough terrain', 'bumpy terrain', 'forbidden terrain', 'obstacle']
        self.merges = [0, 2, 2, 2, 5, 5, 4, 0, 5, 5, 1, 2, 5, 2, 3, 5, 5, 5, 5, 5, 0, 3, 5, 1, 5]
        
        self.sample_count = len(file_list)
        
        RUGD_pics_path = path + 'RUGD_frames-with-annotations/'
        RUGD_annotations_path = path + 'RUGD_annotations/'
        
        self.reset()
        self.annotation_info(RUGD_annotations_path + 'RUGD_annotation-colormap.txt')
        
        self.image_paths = []
        self.annotation_paths = []
        
        self.image_paths = [RUGD_pics_path + f + '.png' for f in file_list]
        self.annotation_paths = [RUGD_annotations_path + f + '.png' for f in file_list]
        
        """
        for directory, subdirlist, filelist in os.walk(RUGD_pics_path):
            
            subdir = directory[len(RUGD_pics_path):]
            
            for f in filelist:
                
                image_path = os.path.join(directory, f)
                annotation_path = os.path.join(RUGD_annotations_path, subdir, f)
                
                self.image_paths.append(image_path)
                self.annotation_paths.append(annotation_path)
        """
                
    def annotation_info(self, filepath):
        
        self.labels = []
        self.pixel_values = []

        with open(filepath, 'r') as file:
            for line in file:
                data = line.strip().split()
                self.labels.append(data[1])
                self.pixel_values.append([int(data[2]), int(data[3]), int(data[4])])
                
        #Merge labels
        labels = [[] for i in range(6)]
        for idx in range(len(self.labels)):
            labels[self.merges[idx]].append(self.labels[idx])
        self.labels = labels
                
        self.class_count = len(self.labels)
        
    def reset(self):
        
        self.idx = 0
        
    def fill_mask_with_color(self, annotation, mask, pixel_color):
        # Get the shape of the annotation
        height, width, _ = annotation.shape
    
        mask+= np.all((annotation == pixel_color), axis = 2).astype(np.uint8)
    
        return mask
        
        
    def get_next(self):
        
        image_path = self.image_paths[self.idx]
        annotation_path = self.annotation_paths[self.idx]
        
        image = imageops.open_image(image_path)
        annotation = imageops.open_image(annotation_path)
        
        shape = annotation.shape
        masks = np.zeros((25, shape[0], shape[1]), dtype=np.uint8)
        
        for i, colour in enumerate(self.pixel_values):
            masks[i] = self.fill_mask_with_color(annotation, masks[i], colour)
        
        def num_or(x, y):
            return 1 - (1 - x) * (1 - y)
        
        #Merge Masks.
        res_mask = np.zeros([6, shape[0], shape[1]], dtype = np.float32)
        for i in range(25):
            midx = self.merges[i]
            res_mask[midx] = num_or(res_mask[midx], masks[i])
        
            
        self.idx += 1
        
        return (image, res_mask)


