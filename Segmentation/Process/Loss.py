"""
====================================================================================================
Package
====================================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Dice
from torchmetrics.classification import MulticlassJaccardIndex


"""
====================================================================================================
Dice
====================================================================================================
"""
def get_dice(labels, predicts):

    dice = Dice(num_classes = 3, average = 'micro')(labels, predicts)

    return dice


"""
====================================================================================================
IoU
====================================================================================================
"""
def get_iou(labels, predicts):

    iou = MulticlassJaccardIndex(num_classes = 3,average = 'micro')(labels, predicts)

    return iou

"""
====================================================================================================
Accuracy
====================================================================================================
"""
def get_acc(labels, predicts):

    correct = torch.sum((labels == predicts), dim = (1, 2))
    total = torch.sum((labels == labels), dim = (1, 2))
    acc = correct / total

    return acc.mean()

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    torch.manual_seed(0)
    inputs = torch.rand((3, 7, 224, 224)).to(torch.int8)
    torch.manual_seed(1)
    target = torch.rand((3, 7, 224, 224)).to(torch.int8)

    dice = get_dice(target, inputs)
    loss = 1- dice
    print()
    print(type(dice), dice)
    print(type(loss), loss)
    print()

    iou = get_iou(target, inputs)
    print()
    print(type(iou), iou)
    print()

    acc = get_acc(target, inputs)
    print()
    print(type(acc), acc)
    print()