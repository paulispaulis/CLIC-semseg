"""
File for image-embedding and caption-embedding datasets.
"""
import pickle as pkl
import numpy as np
import torch
from src import misc
import gc


class EmbDataset():
    def __init__(s):
        s.sample_count = 0
        s.embedding_shape = None
        s.type = None
    
    def get_next(s):
        """
        Function should return a single embedding as a torch tensor.
        """
        raise NotImplemented
        
    def get_batch(s, size = 128):
        res = []
        for sample in range(size):
            res.append(s.get_next())
        res = torch.stack(res, dim = 0)
        return res
        
    def reset(s):
        raise NotImplemented
        
    def __str__(s):
        res = 'EmbDataset of class ' + str(s.__class__.__name__) + '\n'
        res+= 'Sample count: ' + str(s.sample_count) + '\n'
        res+= 'Embedding shape: ' + str(s.embedding_shape) + '\n'
        if s.type == 'image':
            res+= 'Spacial shape: ' + str(s.spacial_shape) + '\n'
        res+= 'Type: ' + s.type + '\n'
        
        return res

    
class RAMLoadDataset(EmbDataset):
    def __init__(s, name, load_amount = 1000, folder = './Data', norm = False):
        """
        Loads an embedding dataset file into RAM.
        Serves up samples as required.
        name - name of file without extension.
        load_amount - number of samples to attempt to load.
            In case enough samples are not available, just loads as much as there are.
        folder - folder path.
        norm - whether to L2 normalize the embeddings after loading.
          This is legacy.
        """
        path = folder + '/' + name + '.pkl'
        file = open(path, 'rb')
        
        #Get header data.
        header = pkl.load(file)
        s.embedding_shape = header['embedding shape']
        s.type = header['type']
        if s.type == 'image':
            s.spacial_shape = header['spacial shape']
            
        #Check we have enough memory.
        if s.type == 'image':
            dpoint_size = s.spacial_shape[0] * s.spacial_shape[1] * s.embedding_shape * 4
        elif s.type == 'caption':
            dpoint_size = s.embedding_shape * 4
        else:
            raise ValueError('The dataset\'s type is ' + str(s.type) + ' I\'m confused.')
        
        required_size = dpoint_size * load_amount * 2
        print('Estimated necessary RAM for loading -', required_size, 'bytes')
        required_size+= 1024 ** 3 #One gig extra, just in case.
        gc.collect()
        if misc.free_ram() < required_size:
            file.close()
            raise Exception('Dataset loading intentionally crashed. Need ~' + str(required_size) + ' Bytes, but have ' + str(misc.free_ram()))
            
        #Get the embedding data.
        s.data = []
        loaded_amount = 0
        while True:
            try:
                obj = pkl.load(file)
                s.data+= obj
                loaded_amount+= len(obj)
            except EOFError:
                break
                
            if loaded_amount >= load_amount:
                break
                
        s.sample_count = loaded_amount
        if loaded_amount > load_amount:
            excess = loaded_amount - load_amount
            s.data = s.data[:-excess]
            s.sample_count = load_amount
        elif loaded_amount < load_amount:
            print('Warning, requested to load', load_amount, 'samples, but dataset contained only', loaded_amount)
            
        s.data = np.stack(s.data, axis = 0)
        gc.collect()
        s.data = torch.tensor(s.data, dtype = torch.float32, device = 'cpu', requires_grad = False)
        if norm:
            gc.collect()
            with torch.no_grad():
                n = torch.norm(s.data, dim = 1, p = 2, keepdim = True)
                s.data = s.data / n
        gc.collect()
        file.close()
        
        s.reset()
        
    
    def get_next(s):
        res = s.data[s.idx]
        s.idx+= 1
        if s.idx == s.sample_count:
            s.idx = 0
        
        return res
    
    def reset(s):
        s.idx = 0
    
    
def prep_text_dataset(dataset, 
                      embedder, 
                      sample_count = 1000, 
                      batch_size = 64, 
                      save_folder = './Data',
                      name = 'untitled'):
    """
    Function that prepares a text-embedding file.
    
    dataset - ImCapDataset object.
    embedder - TextEmbedder object.
    sample_count - amount of samples to run.
    batch_size - size of block in which a batch is saved to the disk.
    save_folder - location relative to notebook, where to save.
    name - name of file (excluding extension).
    """
    import pickle as pkl
    
    #Initiate file.
    path = save_folder + '/' + name + '.pkl'
    file = open(path, 'w')
    file.close()
    file = open(path, 'ab')
    
    #Header object.
    embedding_shape = embedder.embedding_shape
    ds_info = {'embedding shape': embedding_shape, 'sample count': sample_count, 'type': 'caption'}
    pkl.dump(ds_info, file)

    res = []
    def save_current():
        nonlocal res
        pkl.dump(res, file)
        res = []
    
    #Create embeddings.
    dataset.reset()
    for sample in range(sample_count):
        image, caption = dataset.get_next()
        res.append(embedder.forward(caption))
        if len(res) == batch_size:
            save_current()
    if len(res) != 0:
        save_current()
    
    file.close()
    
def prep_image_dataset(dataset,
                       embedder,
                       sample_count = 1000,
                       downsampling = 8,
                       batch_size = 64,
                       save_folder = './Data',
                       name = 'untitled'):
    """
    Function that prepares an embedding-embedding file .
    
    dataset - ImCapDataset object.
    embedder - ImageEmbedder object.
    sample_count - amount of samples to run.
    downsampling - amount of downsampling for image embeddings. 
    batch_size - size of block in which a batch is saved to the disk.
    save_folder - location relative to notebook, where to save.
    name - name of file (excluding extension).
    """
    
    #Initiate file.
    path = save_folder + '/' + name + '.pkl'
    file = open(path, 'w')
    file.close()
    file = open(path, 'ab')
    
    #Header object.
    embedding_shape = embedder.embedding_shape
    spacial_shape = [dim // downsampling for dim in embedder.spacial_shape]
    ds_info = {'embedding shape': embedding_shape, 'spacial shape': spacial_shape, 'sample count': sample_count, 'type': 'image'}
    pkl.dump(ds_info, file)

    res = []
    def save_current():
        nonlocal res
        pkl.dump(res, file)
        res = []
    
    #Create embeddings.
    dataset.reset()
    for sample in range(sample_count):
        image, caption = dataset.get_next()
        res.append(embedder.forward(image)[:, ::downsampling, ::downsampling])
        if len(res) == batch_size:
            save_current()
    if len(res) != 0:
        save_current()
    
    file.close()