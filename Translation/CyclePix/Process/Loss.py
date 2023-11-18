"""
====================================================================================================
Package
====================================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


"""
====================================================================================================
Adversarial Loss
====================================================================================================
"""
def get_adv_loss(predicts, labels):

    adv = MSELoss()(predicts, labels)

    return adv


"""
====================================================================================================
Cycle Consistency Loss
====================================================================================================
"""
def get_cyc_loss(predicts, labels):

    pix = L1Loss()(predicts, labels)

    return pix


"""
====================================================================================================
Pixel-Wise Loss
====================================================================================================
"""
def get_pix_loss(predicts, labels):

    pix = L1Loss()(predicts, labels)

    return pix


"""
====================================================================================================
PSNR
====================================================================================================
"""
def get_psnr(predicts, labels):

    psnr = PeakSignalNoiseRatio()(predicts, labels)

    return psnr

"""
====================================================================================================
SSIM
====================================================================================================
"""
def get_ssim(predicts, labels):

    ssim = StructuralSimilarityIndexMeasure()(predicts, labels)

    return ssim

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    image = torch.rand((2, 1, 512, 512))
    label = torch.rand((2, 1, 512, 512))

    print()
    print(get_adv_loss(image, label))

    print()
    print(get_cyc_loss(image, label))

    print()
    print(get_pix_loss(image, label))

    print()
    print(get_psnr(image, label))

    print()
    print(get_ssim(image, label))
