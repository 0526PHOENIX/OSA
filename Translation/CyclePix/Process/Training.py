"""
====================================================================================================
Package
====================================================================================================
"""
import os
import datetime
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from GAN import Generator, Discriminator
from Loss import get_adv_loss, get_cyc_loss, get_pix_loss
from Loss import get_psnr, get_ssim
from Dataset import Training_2D
from Helper import Buffer


"""
====================================================================================================
Global Constant
====================================================================================================
"""
MAX = 10000000
STRIDE = 5
BATCH = 32
EPOCH = 1

METRICS = 5
METRICS_GEN = 0
METRICS_DIS1 = 1
METRICS_DIS2 = 2
METRICS_PSNR = 3
METRICS_SSIM = 4

DATA_PATH = "C:\\Users\\PHOENIX\\Desktop\\OSA\\TempData"
MODEL_PATH = ""
RESULTS_PATH = "C:\\Users\\PHOENIX\\Desktop\\OSA\\Translation\\CyclePix\\Results"


"""
====================================================================================================
Training
====================================================================================================
"""
class Training():

    """
    ================================================================================================
    Initialize Critical Parameters
    # done
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
    Initialize Model
    # done
    ================================================================================================
    """
    def init_model(self):

        print('\n' + 'Initializing Model' + '\n')

        self.gen12 = Generator().to(self.device)
        self.gen21 = Generator().to(self.device)

        self.dis1 = Discriminator().to(self.device)
        self.dis2 = Discriminator().to(self.device)
    
    """
    ================================================================================================
    Initialize Optimizer
    # done
    ================================================================================================
    """
    def init_optimizer(self):

        print('\n' + 'Initializing Optimizer' + '\n')

        self.optimizer_gen = Adam(chain(self.gen12.parameters(), self.gen21.parameters()), lr = 1e-4)

        self.optimizer_dis1 = Adam(self.dis1.parameters(), lr = 1e-4)
        self.optimizer_dis2 = Adam(self.dis2.parameters(), lr = 1e-4)

    """
    ================================================================================================
    Initialize TensorBorad
    # done
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
    # done
    ================================================================================================
    """
    def init_training_dl(self):

        train_ds = Training_2D(root = DATA_PATH, is_val = False, val_stride = STRIDE)
        train_dl = DataLoader(train_ds, batch_size = BATCH, drop_last = False)

        return train_dl

    """
    ================================================================================================
    Initialize Validation Data Loader
    # done
    ================================================================================================
    """
    def init_validation_dl(self):

        val_ds = Training_2D(root = DATA_PATH, is_val = True, val_stride = STRIDE)
        val_dl = DataLoader(val_ds, batch_size = BATCH, drop_last = False)

        return val_dl
    
    """
    ================================================================================================
    Load Model Parameter and Hyperparameter
    # done
    ================================================================================================
    """
    def load_model(self):

        if os.path.isfile(MODEL_PATH):

            # get checkpoint
            checkpoint = torch.load(MODEL_PATH)
            print('\n' + 'Loading Checkpoint' + '\n')

            # load model
            self.gen12.load_state_dict(checkpoint['gen12_state'])
            self.gen21.load_state_dict(checkpoint['gen21_state'])
            self.dis1.load_state_dict(checkpoint['dis1_state'])
            self.dis2.load_state_dict(checkpoint['dis2_state'])
            print('\n' + 'Loading Model' + '\n')
            
            # load optimizer
            self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state'])
            self.optimizer_dis1.load_state_dict(checkpoint['optimizer_dis1_state'])
            self.optimizer_dis2.load_state_dict(checkpoint['optimizer_dis2_state'])
            print('\n' + 'Loading Optimizer' + '\n')

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
        
            return MAX

    """
    ================================================================================================
    Main Training Function
    # done
    ================================================================================================
    """
    def main(self):

        # data loader
        train_dl = self.init_training_dl()
        val_dl = self.init_validation_dl()

        # load model parameter and get checkpoint
        best_score = self.load_model()

        self.fake1_buffer = Buffer()
        self.fake2_buffer = Buffer()

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
                best_score = min(best_score, score)
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
    # done
    ================================================================================================
    """
    def training(self, epoch_index, train_dl):
        
        # training state
        self.gen12.train()
        self.gen21.train()
        self.dis1.train()   
        self.dis2.train()   

        # create buffer for matrics
        metrics = torch.zeros(METRICS, len(train_dl), device = self.device)

        space = "{:3}{:3}{:3}"
        progress = tqdm(enumerate(train_dl), total = len(train_dl), leave = True,
                        bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
        for batch_index, batch_tuple in progress:

            # get samples
            (real1_t, real2_t) = batch_tuple
            real1_g = real1_t.to(self.device)
            real2_g = real2_t.to(self.device)

            # get output of model
            fake1_g = self.gen21(real2_g)
            fake2_g = self.gen12(real2_g)

            recover1_g = self.gen21(fake2_g)
            recover2_g = self.gen12(fake1_g)

            # ground truth
            valid = torch.ones(real1_g.size(0), 1, 12, 12, requires_grad = False, device = self.device)
            fake = torch.zeros(real2_g.size(0), 1, 12, 12, requires_grad = False, device = self.device)

            """
            ========================================================================================
            Train Generator
            ========================================================================================
            """
            # refresh gradient
            self.optimizer_gen.zero_grad()

            # get pixelwise loss
            loss_pix1 = get_pix_loss(fake2_g, real1_g)
            loss_pix2 = get_pix_loss(fake1_g, real2_g)
            loss_pix = (loss_pix1 + loss_pix2) / 2

            # get adversarial loss
            loss_adv1 = get_adv_loss(self.dis1(fake1_g, real2_g), valid)
            loss_adv2 = get_adv_loss(self.dis2(fake2_g, real1_g), valid)
            loss_adv = (loss_adv1 + loss_adv2) / 2

            # get adversarial loss
            loss_cyc1 = get_cyc_loss(recover1_g, real1_g)
            loss_cyc2 = get_cyc_loss(recover2_g, real2_g)
            loss_cyc = (loss_cyc1 + loss_cyc2) / 2            

            # total loss
            loss_gen = loss_pix + loss_adv + loss_cyc

            # update parameters
            loss_gen.backward()
            self.optimizer_gen.step()

            """
            ========================================================================================
            Train Discriminator 1
            ========================================================================================
            """
            # refresh gradient
            self.optimizer_dis1.zero_grad()

            # real loss
            loss_real1 = get_adv_loss(self.dis1(real1_g, real2_g), valid)

            # fake loss
            fake1_g_ = self.fake1_buffer.push_and_pop(fake1_g)
            loss_fake1 = get_adv_loss(self.dis1(fake1_g_, real2_g), fake)

            # total loss
            loss_dis1 = (loss_real1 + loss_fake1) / 2

            # update parameters
            loss_dis1.backward()
            self.optimizer_dis1.step()
    
            """
            ========================================================================================
            Train Discriminator 2
            ========================================================================================
            """
            # refresh gradient
            self.optimizer_dis1.zero_grad()

            # real loss
            loss_real2 = get_adv_loss(self.dis2(real2_g, real1_g), valid)

            # fake loss
            fake2_g_ = self.fake2_buffer.push_and_pop(fake2_g)
            loss_fake2 = get_adv_loss(self.dis2(fake2_g_, real1_g), fake)

            # total loss
            loss_dis2 = (loss_real2 + loss_fake2) / 2

            # update parameters
            loss_dis2.backward()
            self.optimizer_dis2.step()

            """
            ========================================================================================
            Get and Save Metrics
            ========================================================================================
            """
            # PSNR
            psnr1 = get_psnr(fake1_g, real1_g)
            psnr2 = get_psnr(fake2_g, real2_g)
            psnr = (psnr1 + psnr2) / 2

            # SSIM
            ssim1 = get_ssim(fake1_g, real1_g)
            ssim2 = get_ssim(fake2_g, real2_g)
            ssim = (ssim1 + ssim2) / 2

            # Save Metrics
            metrics[METRICS_GEN, batch_index] = loss_gen
            metrics[METRICS_DIS1, batch_index] = loss_dis1
            metrics[METRICS_DIS2, batch_index] = loss_dis2
            metrics[METRICS_PSNR, batch_index] = psnr
            metrics[METRICS_SSIM, batch_index] = ssim

            progress.set_description('Epoch [' + space.format(epoch_index, ' / ', EPOCH) + ']')
            progress.set_postfix(loss_gen = loss_gen, loss_dis1 = loss_dis1, loss_dis2 = loss_dis2)

        return metrics.to('cpu')

    """
    ================================================================================================
    Validation Loop
    # done
    ================================================================================================
    """
    def validation(self, epoch_index, val_dl):

        with torch.no_grad():

            # validation state
            self.gen12.eval()
            self.gen21.eval()
            self.dis1.eval()   
            self.dis2.eval() 

            # create buffer for matrics
            metrics = torch.zeros(METRICS, len(val_dl), device = self.device)
        
            space = "{:3}{:3}{:3}"
            progress = tqdm(enumerate(val_dl), total = len(val_dl), leave = True,
                            bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
            for batch_index, batch_tuple in progress:

                # get samples
                (real1_t, real2_t) = batch_tuple
                real1_g = real1_t.to(self.device)
                real2_g = real2_t.to(self.device)

                # get output of model
                fake1_g = self.gen21(real2_g)
                fake2_g = self.gen12(real2_g)

                recover1_g = self.gen21(fake2_g)
                recover2_g = self.gen12(fake1_g)

                # ground truth
                valid = torch.ones(real1_g.size(0), 1, 12, 12, requires_grad = False, device = self.device)
                fake = torch.zeros(real2_g.size(0), 1, 12, 12, requires_grad = False, device = self.device)    

                """
                ========================================================================================
                Validate Generator
                ========================================================================================
                """
                # get pixelwise loss
                loss_pix1 = get_pix_loss(fake2_g, real1_g)
                loss_pix2 = get_pix_loss(fake1_g, real2_g)
                loss_pix = (loss_pix1 + loss_pix2) / 2

                # get adversarial loss
                loss_adv1 = get_adv_loss(self.dis1(fake1_g, real2_g), valid)
                loss_adv2 = get_adv_loss(self.dis2(fake2_g, real1_g), valid)
                loss_adv = (loss_adv1 + loss_adv2) / 2

                # get adversarial loss
                loss_cyc1 = get_cyc_loss(recover1_g, real1_g)
                loss_cyc2 = get_cyc_loss(recover2_g, real2_g)
                loss_cyc = (loss_cyc1 + loss_cyc2) / 2            

                # total loss
                loss_gen = loss_pix + loss_adv + loss_cyc

                """
                ========================================================================================
                Validate Discriminator 1
                ========================================================================================
                """
                # real loss
                loss_real1 = get_adv_loss(self.dis1(real1_g, real2_g), valid)

                # fake loss
                fake1_g_ = self.fake1_buffer.push_and_pop(fake1_g)
                loss_fake1 = get_adv_loss(self.dis1(fake1_g_, real2_g), fake)

                # total loss
                loss_dis1 = (loss_real1 + loss_fake1) / 2
        
                """
                ========================================================================================
                Validate Discriminator 2
                ========================================================================================
                """
                # real loss
                loss_real2 = get_adv_loss(self.dis2(real2_g, real1_g), valid)

                # fake loss
                fake2_g_ = self.fake2_buffer.push_and_pop(fake2_g)
                loss_fake2 = get_adv_loss(self.dis2(fake2_g_, real1_g), fake)

                # total loss
                loss_dis2 = (loss_real2 + loss_fake2) / 2

                """
                ========================================================================================
                Get and Save Metrics
                ========================================================================================
                """
                # PSNR
                psnr1 = get_psnr(fake1_g, real1_g)
                psnr2 = get_psnr(fake2_g, real2_g)
                psnr = (psnr1 + psnr2) / 2

                # SSIM
                ssim1 = get_ssim(fake1_g, real1_g)
                ssim2 = get_ssim(fake2_g, real2_g)
                ssim = (ssim1 + ssim2) / 2

                # Save Metrics
                metrics[METRICS_GEN, batch_index] = loss_gen
                metrics[METRICS_DIS1, batch_index] = loss_dis1
                metrics[METRICS_DIS2, batch_index] = loss_dis2
                metrics[METRICS_PSNR, batch_index] = psnr
                metrics[METRICS_SSIM, batch_index] = ssim

                progress.set_description('Epoch [' + space.format(epoch_index, ' / ', EPOCH) + ']')
                progress.set_postfix(loss_gen = loss_gen, loss_dis1 = loss_dis1, loss_dis2 = loss_dis2)

            return metrics.to('cpu')
    
    """
    ================================================================================================
    Save Metrics for Whole Epoch
    # done
    ================================================================================================
    """ 
    def save_metrics(self, epoch_index, mode, metrics_t):

        # copy metrics
        metrics_a = metrics_t.detach().numpy().mean(axis = 1)

        # create a dictionary to save metrics
        metrics_dict = {}
        metrics_dict[mode + '/loss_gen'] = metrics_a[METRICS_GEN]
        metrics_dict[mode + '/loss_dis1'] = metrics_a[METRICS_DIS1]
        metrics_dict[mode + '/loss_dis2'] = metrics_a[METRICS_DIS2]
        metrics_dict[mode + '/psnr'] = metrics_a[METRICS_PSNR]
        metrics_dict[mode + '/ssim'] = metrics_a[METRICS_SSIM]


        # save metrics to tensorboard writer
        writer = getattr(self, mode + '_writer')
        for key, value in metrics_dict.items():

            writer.add_scalar(key, value, epoch_index)
        
        # refresh tensorboard writer
        writer.flush()

        return metrics_dict[mode + '/loss_gen']

    """
    ================================================================================================
    Save Some Image to Checking
    # done
    ================================================================================================
    """ 
    def save_images(self, epoch_index, mode, dataloader):

        # validation state
        self.gen12.eval()
        self.gen21.eval()

        # get random image index and load sample
        (real1_t, real2_t) = dataloader.dataset[60]
        real1_g = real1_t.to(self.device).unsqueeze(0)
        real2_g = real2_t.to(self.device).unsqueeze(0)

        # get predict mask
        fake1_g = self.gen21(real2_g)
        fake1_a = fake1_g.to('cpu').detach().numpy()[0]
        real1_a = real1_g.to('cpu').detach().numpy()[0]

        fake2_g = self.gen12(real1_g)
        fake2_a = fake2_g.to('cpu').detach().numpy()[0]
        real2_a = real2_g.to('cpu').detach().numpy()[0]

        # save image to tensorboard writer
        writer = getattr(self, mode + '_writer')
        writer.add_image(mode + '/real1', real1_a, epoch_index, dataformats = 'CHW')
        writer.add_image(mode + '/fake1', fake1_a, epoch_index, dataformats = 'CHW')
        writer.add_image(mode + '/real2', real2_a, epoch_index, dataformats = 'CHW')
        writer.add_image(mode + '/fake2', fake2_a, epoch_index, dataformats = 'CHW')

        # refresh tensorboard writer
        writer.flush()

    """
    ================================================================================================
    Save Model
    # done
    ================================================================================================
    """ 
    def save_model(self, epoch_index, score, is_best):

        # prepare model state dict
        gen12 = self.gen12
        gen21 = self.gen21
        dis1 = self.dis1
        dis2 = self.dis2

        opt_gen = self.optimizer_gen
        opt_dis1 = self.optimizer_dis1
        opt_dis2 = self.optimizer_dis2

        state = {
            'time': self.time,
            'gen12_state': gen12.state_dict(),
            'gen12_name': type(gen12).__name__,
            'gen21_state': gen21.state_dict(),
            'gen21_name': type(gen21).__name__,
            'dis1_state': dis1.state_dict(),
            'dis1_name': type(dis1).__name__,
            'dis2_state': dis2.state_dict(),
            'dis2_name': type(dis2).__name__,
            'optimizer_gen_state': opt_gen.state_dict(),
            'optimizer_gen_name': type(opt_gen).__name__,
            'optimizer_dis1_state': opt_dis1.state_dict(),
            'optimizer_dis1_name': type(opt_dis1).__name__,
            'optimizer_dis2_state': opt_dis2.state_dict(),
            'optimizer_dis2_name': type(opt_dis2).__name__,
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