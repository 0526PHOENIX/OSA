"""
====================================================================================================
Package
====================================================================================================
"""
import os
import datetime
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Unet import Unet
from Cascade import Cascade
from Loss import get_dice, get_acc, get_iou
from Dataset import Training_2D, Training_3D


"""
====================================================================================================
Global Constant
====================================================================================================
"""
SLICE = 7
STRIDE = 5
CLASS = 3
BATCH = 32
EPOCH = 3

METRICS = 4
METRICS_LOSS = 0
METRICS_ACC = 1
METRICS_DICE = 2
METRICS_IOU = 3

DATA_PATH = "C:\\Users\\PHOENIX\\Desktop\\OSA\\TempData"
MODEL_PATH = ""
RESULTS_PATH = "C:\\Users\\PHOENIX\\Desktop\\OSA\\Segmentation\\Results"


"""
====================================================================================================
Training
====================================================================================================
"""
class Training():

    """
    ================================================================================================
    Initialize Critical Parameters
    ================================================================================================
    """
    def __init__(self):
        
        # training device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print('\n' + 'Training on: ' + str(self.device) + '\n')

        # time and tensorboard writer
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        print('\n' + 'Start From: ' + self.time + '\n')
        self.train_writer = None
        self.val_writer = None

        # model and optimizer
        self.init_model()
        self.init_optimizer()

        # begin epoch
        self.begin = 1

    """
    ================================================================================================
    Initialize Model: Unet
    ================================================================================================
    """
    def init_model(self):

        print('\n' + 'Initializing Model' + '\n')

        self.model = Unet(num_channel = SLICE, num_class = CLASS).to(self.device)
    
    """
    ================================================================================================
    Initialize Optimizer: Adam
    ================================================================================================
    """
    def init_optimizer(self):

        print('\n' + 'Initializing Optimizer' + '\n')

        self.optimizer = Adam(self.model.parameters(), lr = 1e-4)

    """
    ================================================================================================
    Initialize TensorBorad
    ================================================================================================
    """
    def init_tensorboard(self):

        if (self.train_writer == None) or (self.val_writer == None):

            print('\n' + 'Initializing TensorBoard' + '\n')

            # metrics path
            log_dir = os.path.join(RESULTS_PATH, 'Metrics', self.time)

            # training and validation tensorboard writer
            self.train_writer = SummaryWriter(log_dir = (log_dir + '_train'))
            self.val_writer = SummaryWriter(log_dir = (log_dir + '_val'))

    """
    ================================================================================================
    Initialize Training Data Loader
    ================================================================================================
    """
    def init_training_dl(self):

        train_ds = Training_2D(root = DATA_PATH, is_val = False, val_stride = STRIDE, slice = SLICE)
        train_dl = DataLoader(train_ds, batch_size = BATCH, drop_last = False)

        return train_dl

    """
    ================================================================================================
    Initialize Validation Data Loader
    ================================================================================================
    """
    def init_validation_dl(self):

        val_ds = Training_2D(root = DATA_PATH, is_val = True, val_stride = STRIDE, slice = SLICE)
        val_dl = DataLoader(val_ds, batch_size = BATCH, drop_last = False)

        return val_dl
    
    """
    ================================================================================================
    Load Model Parameter and Hyperparameter
    ================================================================================================
    """
    def load_model(self):

        if os.path.isfile(MODEL_PATH):

            # get checkpoint
            checkpoint = torch.load(MODEL_PATH)
            print('\n' + 'Loading Checkpoint' + '\n')

            # load model
            self.model.load_state_dict(checkpoint['model_state'])
            print('\n' + 'Loading Model: ' + checkpoint['model_name'] + '\n')

            # load optimizer
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print('\n' + 'Loading Optimizer: ' + checkpoint['optimizer_name'] + '\n')

            # set time
            self.time = checkpoint['time']
            print('\n' + 'Continued From: ' +  self.time + '\n')

            # set epoch
            if checkpoint['epoch'] < EPOCH:
                self.begin = checkpoint['epoch'] + 1
            else:
                self.begin = 1
            print('\n' + 'Start From Epoch: ' + str(self.begin) + '\n')

            # set tensorboard
            log_dir = os.path.join(RESULTS_PATH, 'Metrics', checkpoint['time'])
            self.train_writer = SummaryWriter(log_dir = (log_dir + '_train'))
            self.val_writer = SummaryWriter(log_dir = (log_dir + '_val'))

            return checkpoint['score']
        
        else:

            self.init_tensorboard()
        
            return 0.0

    """
    ================================================================================================
    Main Training Function
    ================================================================================================
    """
    def main(self):

        # data loader
        train_dl = self.init_training_dl()
        val_dl = self.init_validation_dl()

        # load model parameter and get checkpoint
        best_score = self.load_model()

        # main loop
        count = 0
        for epoch_index in range(self.begin, EPOCH + 1):
            
            # get and save metrics
            print('Training: ')
            metrics_train = self.training(epoch_index, train_dl)
            self.save_metrics(epoch_index, 'train', metrics_train)

            # check performance for every 10 epochs
            if epoch_index == 1 or epoch_index % STRIDE == 0:
                
                # get and save metrics
                print('\n' + 'Validation: ')
                metrics_val = self.validation(epoch_index, val_dl)
                score = self.save_metrics(epoch_index, 'val', metrics_val)

                # save model
                best_score = max(best_score, score)
                self.save_model(epoch_index, score, (best_score == score))

                # save image for checking
                self.save_images(epoch_index, 'train', train_dl)
                self.save_images(epoch_index, 'val', val_dl)

                # early stop
                if score == best_score:
                    count = 0
                elif count < 5:
                    count += 1
                elif count == 5:
                    print('\n' + 'early stop' + '\n')
                    break

                print()
        
        self.train_writer.close()
        self.val_writer.close()

    """
    ================================================================================================
    Training Loop
    ================================================================================================
    """
    def training(self, epoch_index, train_dl):
        
        # training state
        self.model.train()

        # create buffer for matrics
        metrics = torch.zeros(METRICS, len(train_dl), device = self.device)

        space = "{:3}{:3}{:3}"
        progress = tqdm(enumerate(train_dl), total = len(train_dl), leave = True,
                        bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
        for batch_index, batch_tuple in progress:

            # get samples
            (images_t, labels_t) = batch_tuple
            images_g = images_t.to(self.device)
            labels_g = labels_t.to(self.device)

            # get output of model
            predicts_g = self.model(images_g)

            # refresh gradient
            self.optimizer.zero_grad()

            # get loss and metrics
            dice = get_dice(predicts_g, labels_g)
            loss = torch.ones(1, requires_grad = True, device = self.device) - dice

            # update parameters
            loss.backward()
            self.optimizer.step()

            # get ACC and IoU
            acc = get_acc(predicts_g, labels_g)
            iou = get_iou(predicts_g, labels_g)

            # save loss value and matrics to buffer
            metrics[METRICS_LOSS, batch_index] = loss
            metrics[METRICS_ACC, batch_index] = acc
            metrics[METRICS_DICE, batch_index] = dice
            metrics[METRICS_IOU, batch_index] = iou

            progress.set_description('Epoch [' + space.format(epoch_index, ' / ', EPOCH) + ']')
            progress.set_postfix(loss = loss.item(), acc = acc.item())

        return metrics.to('cpu')

    """
    ================================================================================================
    Validation Loop
    ================================================================================================
    """
    def validation(self, epoch_index, val_dl):

        with torch.no_grad():

            # validation state
            self.model.eval()

            # create buffer for matrics
            metrics = torch.zeros(METRICS, len(val_dl), device = self.device)
        
            space = "{:3}{:3}{:3}"
            progress = tqdm(enumerate(val_dl), total = len(val_dl), leave = True,
                            bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
            for batch_index, batch_tuple in progress:

                # get samples
                (images_t, labels_t) = batch_tuple
                images_g = images_t.to(self.device)
                labels_g = labels_t.to(self.device)

                # get output of model
                predicts_g = self.model(images_g)

                # get loss and metrics
                dice = get_dice(predicts_g, labels_g)
                loss = torch.ones(1, requires_grad = True, device = self.device) - dice

                # get ACC and IoU
                acc = get_acc(predicts_g, labels_g)
                iou = get_iou(predicts_g, labels_g)

                # save loss value and matrics to buffer
                metrics[METRICS_LOSS, batch_index] = loss
                metrics[METRICS_ACC, batch_index] = acc
                metrics[METRICS_DICE, batch_index] = dice
                metrics[METRICS_IOU, batch_index] = iou

                progress.set_description('Epoch [' + space.format(epoch_index, '/', EPOCH) + ']')
                progress.set_postfix(val_loss = loss.item(), val_acc = acc.item())

        return metrics.to('cpu')
    
    """
    ================================================================================================
    Save Metrics for Whole Epoch
    ================================================================================================
    """ 
    def save_metrics(self, epoch_index, mode, metrics_t):

        # copy metrics
        metrics_a = metrics_t.detach().numpy().mean(axis = 1)

        # create a dictionary to save metrics
        metrics_dict = {}
        metrics_dict[mode + '/loss'] = metrics_a[METRICS_LOSS]
        metrics_dict[mode + '/acc'] = metrics_a[METRICS_ACC]
        metrics_dict[mode + '/dice'] = metrics_a[METRICS_DICE]
        metrics_dict[mode + '/iou'] = metrics_a[METRICS_IOU]

        # save metrics to tensorboard writer
        writer = getattr(self, mode + '_writer')
        for key, value in metrics_dict.items():

            writer.add_scalar(key, value, epoch_index)
        
        # refresh tensorboard writer
        writer.flush()

        return metrics_dict[mode + '/dice']

    """
    ================================================================================================
    Save Some Image to Checking
    ================================================================================================
    """ 
    def save_images(self, epoch_index, mode, dataloader):

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
        image_a = np.zeros((3, shape[1], shape[2]), dtype = np.float32)
        image_a[:, :, :] = image_t.to('cpu').detach().numpy()[SLICE // 2, :, :]
        image_a[0, :, :] += (predict_a & label_a)[0, :, :]
        image_a /= 3
        image_a.clip(0, 1, image_a)

        # save image to tensorboard writer
        writer = getattr(self, mode + '_writer')
        writer.add_image(mode + '/predict', image_a, epoch_index, dataformats = 'CHW')

        """
        ============================================================================================
        Label Heatmap
        ============================================================================================
        """   
        if epoch_index == 1:
                
            # get heatmap
            image_a = np.zeros((3, shape[1], shape[2]), dtype = np.float32)
            image_a[:, :, :] = image_t.to('cpu').detach().numpy()[SLICE // 2, :, :]
            image_a[0, :, :] += label_a[0, :, :]

            image_a /= 2
            image_a.clip(0, 1, image_a)

            # save image to tensorboard writer
            writer = getattr(self, mode + '_writer')
            writer.add_image(mode + '/label', image_a, epoch_index, dataformats = 'CHW')

        # refresh tensorboard writer
        writer.flush()

    """
    ================================================================================================
    Save Model
    ================================================================================================
    """ 
    def save_model(self, epoch_index, score, is_best):

        # prepare model state dict
        model = self.model
        opt = self.optimizer
        state = {
            'time': self.time,
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': opt.state_dict(),
            'optimizer_name': type(opt).__name__,
            'epoch': epoch_index,
            'score': score,
        }

        # save model
        model_path = os.path.join(RESULTS_PATH, 'Model', self.time + '_epoch' + str(epoch_index) + '.pt')
        torch.save(state, model_path)

        if is_best:

            # save best model
            best_path = os.path.join(RESULTS_PATH, 'Model', self.time + '.best.pt')
            torch.save(state, best_path)


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    Training().main()