import os
import numpy as np
import SimpleITK as sitk

import torch
from torch import tensor
from torchmetrics.classification import Dice
import datetime


MODEL_PATH = "C:\\Users\\PHOENIX\\Desktop\\OSA_Project\\Segmentation\\Results\\Model\\2023-09-29_01.00_1.best.pt"

filename = os.path.basename(MODEL_PATH)

print(filename)