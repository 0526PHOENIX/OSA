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
def get_dice(predicts, labels):

    dice = Dice(num_classes = 3, average = 'micro')(predicts, labels)

    return dice


"""
====================================================================================================
IoU
====================================================================================================
"""
def get_iou(predicts, labels):

    iou = MulticlassJaccardIndex(num_classes = 3, average = 'micro')(predicts, labels)

    return iou

"""
====================================================================================================
Accuracy
====================================================================================================
"""
def get_acc(predicts, labels):

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

    dice = get_dice(inputs, target)
    print(dice.size(0))
    loss = 1- dice
    print()
    print(type(dice), dice)
    print(type(loss), loss)
    print()