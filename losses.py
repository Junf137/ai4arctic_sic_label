import torch
from torch import nn


class OrderedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(OrderedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        loss = criterion(output, target)
        # calculate the hard predictions by using softmax followed by an argmax
        softmax = torch.nn.functional.softmax(output, dim=1)
        hard_prediction = torch.argmax(softmax, dim=1)
        # set the mask according to ignore index
        mask = target == self.ignore_index
        hard_prediction = hard_prediction[~mask]
        target = target[~mask]
        # calculate the absolute difference between target and prediction
        weights = torch.abs(hard_prediction-target) + 1
        # remove ignored index losses
        loss = loss[~mask]
        # if done normalization with weights the loss becomes of the order 1e-5
        # loss = (loss * weights)/weights.sum()
        loss = (loss * weights)
        loss = loss.mean()

        return loss
