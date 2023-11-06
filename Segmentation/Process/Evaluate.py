"""
====================================================================================================
Package
====================================================================================================
"""
import os
import datetime
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Unet import Unet
from Cascade import Cascade
from Loss import get_dice, get_acc, get_iou
from Dataset import Testing_2D, Testing_3D


"""
====================================================================================================
Global Constant
====================================================================================================
"""
SLICE = 7
CLASS = 3
BATCH = 32

METRICS = 4
METRICS_LOSS = 0
METRICS_ACC = 1
METRICS_DICE = 2
METRICS_IOU = 3

DATA_PATH = "C:\\Users\\PHOENIX\\Desktop\\OSA_Project\\TempData"
MODEL_PATH = "C:\\Users\\PHOENIX\\Desktop\\OSA_Project\\Segmentation\\Results\\Model\\2023-09-29_04.09.best.pt"
RESULTS_PATH = "C:\\Users\\PHOENIX\\Desktop\\OSA_Project\\Segmentation\\Results"


"""
====================================================================================================
Evaluate
====================================================================================================
"""
class Evaluate():

    """
    ================================================================================================
    Initialize Critical Parameters
    ================================================================================================
    """
    def __init__(self):

        # evaluating device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print('\n' + 'Evaluating on: ' + str(self.device) + '\n')

        # time and tensorboard writer
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')
        print('\n' + self.time + '\n')
        self.test_writer = None

        # model
        self.model = self.init_model()

    """
    ================================================================================================
    Initialize Model: Unet
    ================================================================================================
    """
    def init_model(self):

        print('\n' + 'Initializing Model' + '\n')

        model = Unet(num_channel = SLICE, num_class = CLASS).to(self.device)

        return model

    """
    ================================================================================================
    Initialize TensorBorad
    ================================================================================================
    """  
    def init_tensorboard(self):

        if self.test_writer == None:

            print('\n' + 'Initializing TensorBoard' + '\n')

            # metrics path
            log_dir = os.path.join(RESULTS_PATH, 'Metrics', self.time)

            # testing tensorboard writer
            self.test_writer = SummaryWriter(log_dir = (log_dir + '_test'))
    
    """
    ================================================================================================
    Initialize Testing Data Loader
    ================================================================================================
    """
    def init_testing_dl(self):

        test_ds = Testing_2D(root = DATA_PATH, slice = SLICE)
        test_dl = DataLoader(test_ds, batch_size = BATCH, drop_last = False)

        return test_dl

    """
    ================================================================================================
    Load Model Parameter
    ================================================================================================
    """
    def load_model(self):

        if os.path.isfile(MODEL_PATH):

            # get checkpoint
            checkpoint = torch.load(MODEL_PATH)
            print('\n' + 'Loading checkpoint' + '\n')

            # load model
            self.model.load_state_dict(checkpoint['model_state'])
            print('\n' + 'Loading model: ' + checkpoint['model_name'] + '\n')

    """
    ================================================================================================
    Main Evaluating Function
    ================================================================================================
    """
    def main(self, file_path = ""):

        # dataloader
        test_dl = self.init_testing_dl()

        # initialize tensorboard and load model parameter
        self.init_tensorboard()
        self.load_model()

        # evaluate model with test set
        print('\n' + 'Testing: ')
        metrics_test = self.testing(test_dl)
        self.save_metrics(metrics_test)
        self.save_images(test_dl)

        # inference with new data
        print('\n' + 'Inference: ')
        self.inference(file_path)

    """
    ================================================================================================
    Testing Loop
    ================================================================================================
    """
    def testing(self, test_dl):

        # validation state
        self.model.eval()

        # create buffer for matrics
        metrics = torch.zeros(METRICS, len(test_dl), device = self.device)
    
        progress = tqdm(enumerate(test_dl), total = len(test_dl), leave = True,
                        bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
        for batch_index, batch_tuple in progress:

            # get metrics
            self.get_result(batch_index, batch_tuple, metrics)
            progress.set_description('Evaluating')
            progress.set_postfix(test_loss = metrics[METRICS_LOSS, batch_index],
                                    test_acc = metrics[METRICS_ACC, batch_index])

        return metrics.to('cpu')

    """
    ================================================================================================
    Get Result: Dice Loss + Accuracy + Dice + IOU
    ================================================================================================
    """
    def get_result(self, batch_index, batch_tuple, metrics):

        # get samples
        (images_t, labels_t) = batch_tuple
        images_g = images_t.to(self.device)
        labels_g = labels_t.to(self.device)

        # get output of model
        predicts_g = self.model(images_g)

        # compute loss value and matrics
        loss, dice = self.get_loss(labels_g, predicts_g)
        acc, iou = self.get_metrics(labels_g, predicts_g)

        # save loss value and matrics to buffer
        metrics[METRICS_LOSS, batch_index] = loss
        metrics[METRICS_ACC, batch_index] = acc
        metrics[METRICS_DICE, batch_index] = dice
        metrics[METRICS_IOU, batch_index] = iou

    """
    ================================================================================================
    Get Loss: Dice Loss + Dice
    ================================================================================================
    """
    def get_loss(self, labels, predicts):

        with torch.no_grad():

            # dice
            dice = get_dice(labels, predicts)
            # dice loss
            loss = 1 - dice

        return (loss, dice)
    
    """
    ================================================================================================
    Get Metrics: Accuracy + IOU
    ================================================================================================
    """
    def get_metrics(self, labels, predicts):

        with torch.no_grad():
            
            # accuracy
            acc = get_acc(labels, predicts)
            # intersection of union
            iou = get_iou(labels, predicts)

        return (acc, iou)

    """
    ================================================================================================
    Save Metrics for Whole Epoch
    ================================================================================================
    """ 
    def save_metrics(self, metrics_t):

        # copy metrics
        metrics_a = metrics_t.detach().numpy().mean(axis = 1)

        # create a dictionary to save metrics
        metrics_dict = {}
        metrics_dict['test/loss'] = metrics_a[METRICS_LOSS]
        metrics_dict['test/acc'] = metrics_a[METRICS_ACC]
        metrics_dict['test/dice'] = metrics_a[METRICS_DICE]
        metrics_dict['test/iou'] = metrics_a[METRICS_IOU]

        # save metrics to tensorboard writer
        for key, value in metrics_dict.items():

            self.test_writer.add_scalar(key, value)
        
        # refresh tensorboard writer
        self.test_writer.flush()

    """
    ================================================================================================
    Save Some Image to Checking
    ================================================================================================
    """ 
    def save_images(self, dataloader):

        self.model.eval()

        # get random image index and load sample
        (image_t, label_t) = dataloader.dataset[60]
        image_g = image_t.to(self.device).unsqueeze(0)
        label_g = label_t.to(self.device).unsqueeze(0)

        # get predict mask
        predict_g = self.model(image_g)
        predict_a = predict_g.to('cpu').detach().numpy()[0]
        label_a = label_g.to('cpu').detach().numpy()[0]

        # shape of image
        shape = label_a.shape

        """
        ============================================================================================
        Predict Heatmap
        ============================================================================================
        """ 
        # get heatmap
        shape = label_a.shape
        image_a = np.zeros((3, shape[1], shape[2]), dtype = np.float32)
        image_a[:, :, :] = image_t.to('cpu').detach().numpy()[SLICE // 2, :, :]
        image_a[0, :, :] += (predict_a & label_a)[0, :, :]
        image_a /= 3
        image_a.clip(0, 1, image_a)

        # save image to tensorboard writer
        self.test_writer.add_image('test/predict', image_a, dataformats = 'CHW')

        """
        ============================================================================================
        Label Heatmap
        ============================================================================================
        """   
        # get heatmap
        image_a = np.zeros((3, shape[1], shape[2]), dtype = np.float32)
        image_a[:, :, :] = image_t.to('cpu').detach().numpy()[SLICE // 2, :, :]
        image_a[0, :, :] += label_a[0, :, :]

        image_a /= 2
        image_a.clip(0, 1, image_a)

        # save image to tensorboard writer
        self.test_writer.add_image('test/label', image_a, dataformats = 'CHW')

        # refresh tensorboard writer
        self.test_writer.flush()
    
    """
    ================================================================================================
    Inference
    ================================================================================================
    """
    def inference(self, file_path):

        self.model.eval()

        # read image to tensor
        series = sitk.ReadImage(file_path)
        series = np.array(sitk.GetArrayFromImage(series), dtype = np.float32)
        series = torch.from_numpy(series).to(torch.float32)

        # preprocessing
        series -= series.min()
        series /= series.max()

        # critical parameter
        width = SLICE // 2
        num_slices = series.size(0) - (width + 1) * 2

        # get predict mask
        mask_g = torch.zeros((num_slices, series.size(1), series.size(2)), device = self.device)
        for i in range(num_slices):

            slice = i + width + 1
            image_t = series[slice - width : slice + width + 1, :, :]
            image_g = image_t.unsqueeze(0).to(self.device)

            mask_g[i, :, :] = self.model(image_g)
        
        # convert mask to image
        mask_t = mask_g.to('cpu').detach().numpy()
        mask = sitk.GetImageFromArray(mask_t)

        # save as .nii.gz
        mask_path = os.path.join(RESULTS_PATH, 'Mask', os.path.basename(file_path))
        sitk.WriteImage(mask, mask_path)
        print('Save Mask to: ' + '"' + mask_path + '\n' + '"')


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    file_path = "C:\\Users\\PHOENIX\\Desktop\\OSA_Project\\TempData\\imagesTr\\2PD64XHB.nii"

    Evaluate().main(file_path)