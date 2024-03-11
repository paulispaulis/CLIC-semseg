"""
File for defining metrics.
"""

import numpy as np
import torch
import torch.nn.functional as f

import src.losses as losses
import src.tensorconversions as tensorconversions
import src.defdevice as defdevice

class Accumulator():
    """
    Abstract class for objects that accumulate metric intermediate results.
    """
    
    def __init__(s):
        pass
    
    def add_sample(s, correct, predicted):
        """
        correct - numpy array of shape [m, y, x]. Binary ground truth masks.
          where m is the number of mask types.
        predicted - numpy array of shape [m, y, x]. Predicted probabilities for masks.
        """
        raise NotImplemented()
    
    def calculate(s):
        """
        Function should calculate and return final metric value/s. 
        """
        raise NotImplemented()
    
        
class PixelSE(Accumulator):
    """
    For calculating mean squared error over all pixels and masks.
    """
    
    def __init__(s):
        super().__init__()
        s.pixel_count = 0
        s.cumulative_se = 0.0
    
    def add_sample(s, correct, predicted):
        s.pixel_count+= correct.size
        s.cumulative_se+= ((correct - predicted) ** 2).sum()
    
    def calculate(s):
        s.result = s.cumulative_se / s.pixel_count
        return s.result


class PixelSECategory(Accumulator):
    """
    For calculating mean squared error over each mask type seperately.
    """
    
    def __init__(s, mask_count):
        super().__init__()
        s.pixel_count = 0
        s.cumulative_se = np.zeros([mask_count])
    
    def add_sample(s, correct, predicted):
        s.pixel_count+= correct[0].size
        s.cumulative_se+= ((correct - predicted) ** 2).sum(axis = 2).sum(axis = 1)
    
    def calculate(s):
        s.result = s.cumulative_se / s.pixel_count
        return s.result
    
class ConfusionMatrix(Accumulator):
    """
    For calculating TP, FP, TN, FN for each category.
    Assumes ground truth masks are exclusive and exhaustive.
    """
    
    def __init__(s, mask_count):
        super().__init__()
        s.pixel_count = 0
        s.TP = np.zeros([mask_count])
        s.FP = np.zeros([mask_count])
        s.TN = np.zeros([mask_count])
        s.FN = np.zeros([mask_count])
    
    def add_sample(s, correct, predicted):
        s.pixel_count+= correct[0].size
        
        correct_idx = correct.argmax(axis = 0)
        predicted_idx = predicted.argmax(axis = 0)
        
        predicted_one_hot = np.zeros(correct.shape)
        indices_y, indices_x = np.indices((correct.shape[1], correct.shape[2]))
        predicted_one_hot[predicted_idx, indices_y, indices_x] = 1
        
        s.TP+= ((correct == 1) & (predicted_one_hot == 1)).sum(axis = 2).sum(axis = 1)
        s.FP+= ((correct == 0) & (predicted_one_hot == 1)).sum(axis = 2).sum(axis = 1)
        s.TN+= ((correct == 0) & (predicted_one_hot == 0)).sum(axis = 2).sum(axis = 1)
        s.FN+= ((correct == 1) & (predicted_one_hot == 0)).sum(axis = 2).sum(axis = 1)
        
    def calculate(s):
        s.result = {'pix count': s.pixel_count, 'TP': s.TP, 'FP': s.FP, 'TN': s.TN, 'FN': s.FN}
        return s.result
    

def accuracy(cmtx):
    """
    Calculates pixel accuracy.
    cmtx - ConfusionMatrix object.
    """
    res = cmtx.TP.sum() / cmtx.pixel_count
    return res

def accuracy_category(cmtx):
    """
    Calculates pixel accuracy by category.
    cmtx - ConfusionMatrix object.
    """
    res = (cmtx.TP + cmtx.TN) / cmtx.pixel_count
    return res

def recall_category(cmtx):
    """
    Calculates what percentage of class is correctly identified.
    cmtx - ConfusionMatrix object.
    """
    res = (cmtx.TP) / (cmtx.TP + cmtx.FN + 0.00001)
    return res

def recall_mean(cmtx):
    """
    Calculates mean recall over categories.
    cmtx - ConfusionMatrix object.
    """
    recalls = recall_category(cmtx)
    return recalls.mean()

def precision_category(cmtx):
    """
    Calculates what percentage of positive guesses within class are correct.
    cmtx - ConfusionMatrix object.
    """
    res = (cmtx.TP) / (cmtx.TP + cmtx.FP + 0.00001)
    return res

def precision_mean(cmtx):
    """
    Calculates mean recall over categories.
    cmtx - ConfusionMatrix object.
    """
    precisions = precision_category(cmtx)
    return precisions.mean()

def IoU_category(cmtx):
    """
    Calculates intersection over union.
    Followed the math at https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU
    Unsure how they deal with the case where all samples in a category are true negatives.
      In this case this function defaults to IoU = 0.
      
    cmtx - ConfusionMatrix object.
    """
    
    IoU = (cmtx.TP) / (cmtx.TP + cmtx.FN + cmtx.FP + 0.00001)
    return IoU

def IoU_mean(cmtx):
    """
    Calculates meanIoU.
    Followed the math at https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU
    
    cmtx - ConfusionMatrix object.
    """
    IoUs = IoU_category(cmtx)
    return IoUs.mean()

def F1_category(cmtx):
    r = recall_category(cmtx)
    p = precision_category(cmtx)
    res = 2 / (r + p)
    return res

def F1_mean(cmtx):
    Fs = F1_category(cmtx)
    return Fs.mean()

def calc_metrics(labels, cmtx = None, se_tracker = None):
    """
    Calculates all metrics.
    labels - list of string; labels of segmentation.
    cmtx - ConfusionMatrix object.
    se_tracker - PixelSE object.
    
    Returns tuple (res_str, res_dict)
    res_str - all metrics formatted as a string.
    res_dict - all metrics stored in a dict.
    """
    
    res = ''
    res_dict = {}
    
    if se_tracker is not None:
        res+= 'MSE over everything - ' + str(se_tracker.calculate()) + '\n'
    
    if cmtx is not None:
        cmtx.calculate()

        tmp = accuracy(cmtx)
        res+= 'Pixel accuracy - ' + str(tmp) + '\n'
        res_dict['Accuracy'] = tmp

        tmp = recall_mean(cmtx)
        res+= 'Mean recall - ' + str(tmp) + '\n'
        res_dict['Recall Mean'] = tmp

        tmp = precision_mean(cmtx)
        res+= 'Mean precision - ' + str(tmp) + '\n'
        res_dict['Precision Mean'] = tmp

        tmp = IoU_mean(cmtx)
        res+= 'Mean IoU - ' + str(tmp) + '\n'
        res_dict['IoU Mean'] = tmp

        tmp = F1_mean(cmtx)
        res+= 'Mean F1 - ' + str(tmp) + '\n'
        res_dict['F1 Mean'] = tmp


        def minmax(values, labels, tag, amount = 5):
            pairs = [{'name': labels[idx], 'val': values[idx]} for idx in range(len(labels))]
            pairs.sort(key = lambda x: x['val'])

            res = '\n'
            for idx in range(min(5, len(labels))):
                res+= 'Small ' + str(tag) + ': ' + str(pairs[idx]['val']) + ' - ' + str(pairs[idx]['name']) + '\n'

            res+= '\n'
            for idx in range(min(5, len(labels))):
                res+= 'Big   ' + str(tag) + ': ' + str(pairs[-idx - 1]['val']) + ' - ' + str(pairs[-idx - 1]['name']) + '\n'

            return res

        ##I think this metric isn't very useful in this case.
        #tmp = accuracy_category(cmtx)
        #res+= minmax(tmp, dataset.labels, 'Accuracy')
        #res_dict['Accruacy Category'] = tmp

        tmp = recall_category(cmtx)
        res+= minmax(tmp, labels, 'Recall')
        res_dict['Recall Category'] = tmp

        tmp = precision_category(cmtx)
        res+= minmax(tmp, labels, 'Precision')
        res_dict['Precision Category'] = tmp

        tmp = IoU_category(cmtx)
        res+= minmax(tmp, labels, 'IoU')
        res_dict['IoU Category'] = tmp

        tmp = F1_category(cmtx)
        res+= minmax(tmp, labels, 'F1')
        res_dict['F1 Category'] = tmp

    return (res, res_dict)    

def test_segmodel(model, dataset, amount = 100, multilabel = False, aggregation = 'max'):
    """
    Function that runs all the test.
    model - segmodel.SegModel object.
    dataset - ImMaskDataset object.
    amount - amount of images to run.
    multilabel - Just use True if using RUGD6.
    """
    
    cmtx = ConfusionMatrix(dataset.class_count)
    se_tracker = PixelSE()
    
    dataset.reset()
    
    #Run tests
    labels = dataset.labels
    for sample_idx in range(amount):
        image, mask = dataset.get_next()
        
        if not multilabel:
            probs = model.forward(image, labels)
        else:
            probs = model.forward_multilabel(image, labels, aggregation = aggregation)
        
        cmtx.add_sample(correct = mask, predicted = probs)
        se_tracker.add_sample(correct = mask, predicted = probs)
        
    res, res_dict = calc_metrics(dataset.labels, cmtx, se_tracker)

    return (res, res_dict)    

def test(model, dataset, image_embedder, text_embedder, amount = 100):
    """
    Function that runs all the tests.
    dataset - ImMaskDataset object.
    text_embedder - TextEmbedder object.
    image_embedder - ImageEmbedder object.
    
    Returns tuple (text output, results)
      text output - test results formatted in a string.
      results - dictionary containing all results.
    """
    
    cmtx = ConfusionMatrix(dataset.class_count)
    se_tracker = PixelSE()
    
    dataset.reset()
    
    #Get label embeddings
    embeddings = []
    for label in dataset.labels:
        embeddings.append(text_embedder.forward(label))
    embeddings = np.stack(embeddings, axis = 0)
    with torch.no_grad():
        embeddings = torch.tensor(embeddings, requires_grad = False, device = defdevice.def_device)
        embeddings = model.forward(embeddings)
    
    #Run tests
    for sample_idx in range(amount):
        image, mask = dataset.get_next()
        
        image_emb = image_embedder.forward(image)
        image_emb = torch.tensor(image_emb, requires_grad = False, device = defdevice.def_device)
        
        prods = torch.einsum('eyx,ne->nyx', image_emb, embeddings)
        probs = torch.sigmoid(prods)
        probs = probs.unsqueeze(0)
        probs = f.interpolate(probs, size = mask.shape[1:], mode = 'bilinear')
        probs = probs[0]
        probs = tensorconversions.tnp(probs)
        
        cmtx.add_sample(correct = mask, predicted = probs)
        se_tracker.add_sample(correct = mask, predicted = probs)
        
    res, res_dict = calc_metrics(dataset.labels, cmtx, se_tracker)

    return (res, res_dict)