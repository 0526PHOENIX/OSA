"""
====================================================================================================
Package
====================================================================================================
"""
import os
import random
import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset


"""
====================================================================================================
2D Training Dataset
(Z, X, Y) = (7, 192, 192)
====================================================================================================
"""
class Training_2D(Dataset):

    def __init__(self, root = "", is_val = False, val_stride = 10):

        # file path setting
        self.root = root
        self.images_path = os.path.join(self.root, 'imagesTr')
        self.labels_path = os.path.join(self.root, 'labelsTr')

        # load image path into list
        self.images = []
        for series in sorted(os.listdir(self.images_path)):
            self.images.append(os.path.join(self.images_path, series))

        # load label path into list
        self.labels = []
        for series in sorted(os.listdir(self.labels_path)):
            self.labels.append(os.path.join(self.labels_path, series))

        # split training and validation
        if is_val:
            self.images = self.images[::val_stride]
            self.labels = self.labels[::val_stride]
        else:
            del self.images[::val_stride]
            del self.labels[::val_stride]

        # shuffle data
        samples = list(zip(self.images, self.labels))
        random.shuffle(samples)
        (self.images, self.labels) = zip(*samples)

        # check data quantity
        if len(self.images) != len(self.labels):
            raise ValueError('Unequal amount of images and labels.')
        
        # 2D slice parameter
        self.num_slices = 192                                       # slices per series
        self.num_series = len(self.images)                          # number of series
        
    def __len__(self):
        
        return (self.num_series * self.num_slices)

    def __getitem__(self, index):
        
        # index
        series_index = index // self.num_slices                     # which series
        slices_index = index % self.num_slices                      # which slice

        # get image data
        image = sitk.ReadImage(self.images[series_index])
        image = np.array(sitk.GetArrayFromImage(image), dtype = np.float32)
        image = torch.from_numpy(image).to(torch.float32)
        image = image[slices_index, :, :].unsqueeze(0)
        
        # get label data
        label = sitk.ReadImage(self.labels[series_index])
        label = np.array(sitk.GetArrayFromImage(label), dtype = np.float32)
        label = torch.from_numpy(label).to(torch.float32)
        label = label[slices_index, :, :].unsqueeze(0)

        return (image, label)


"""
====================================================================================================
2D Testing Dataset
(Z, X, Y) = (7, 192, 192)
====================================================================================================
"""
class Testing_2D(Dataset):

    def __init__(self, root = ""):

        # file path setting
        self.root = root
        self.images_path = os.path.join(self.root, 'imagesTs')
        self.labels_path = os.path.join(self.root, 'labelsTs')

        # load image path into list
        self.images = []
        for series in sorted(os.listdir(self.images_path)):
            self.images.append(os.path.join(self.images_path, series))

        # load label path into list
        self.labels = []
        for series in sorted(os.listdir(self.labels_path)):
            self.labels.append(os.path.join(self.labels_path, series))

        # shuffle data
        samples = list(zip(self.images, self.labels))
        random.shuffle(samples)
        (self.images, self.labels) = zip(*samples)

        # check data quantity
        if len(self.images) != len(self.labels):
            raise ValueError('Unequal amount of images and labels.')
        
        # 2D slice parameter
        self.num_slices = 192                                       # slices per series
        self.num_series = len(self.images)                          # number of series

    def __len__(self):

        return (self.num_series * self.num_slices)

    def __getitem__(self, index):

        # index
        series_index = index // self.num_slices                     # which series
        slices_index = index % self.num_slices                      # which slice

        # get image data
        image = sitk.ReadImage(self.images[series_index])
        image = np.array(sitk.GetArrayFromImage(image), dtype = np.float32)
        image = torch.from_numpy(image).to(torch.float32)
        image = image[slices_index, :, :].unsqueeze(0)
        
        # get label data
        label = sitk.ReadImage(self.labels[series_index])
        label = np.array(sitk.GetArrayFromImage(label), dtype = np.float32)
        label = torch.from_numpy(label).to(torch.float32)
        label = label[slices_index, :, :].unsqueeze(0)

        return (image, label)
    

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    filepath = "C:\\Users\\PHOENIX\\Desktop\\OSA\\TempData"

    train_2D = Training_2D(filepath, False, 10)
    print()
    print(len(train_2D))
    image, label = train_2D[155]
    print(type(image), image.device, image.shape, image.max(), image.min())
    print(type(label), label.device, label.shape, label.unique())
    print()


    # test_2D = Training_2D(filepath, False, 10, 7)
    # print()
    # print(len(test_2D))
    # image, label = test_2D[50]
    # print(type(image), image.device, image.shape, image.max(), image.min())
    # print(type(label), label.device, label.shape, label.unique())
    # print()

