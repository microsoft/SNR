# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import ceil, floor

class softCrossEntropy(nn.Module):
    def __init__(self, ignore_index= -1):
        super(softCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions (N, C, H, W)
        :param target: target distribution (N, C, H, W)
        :return: loss
        """
        assert inputs.size() == target.size()
        mask = (target != self.ignore_index)

        log_likelihood = F.log_softmax(inputs, dim=1)
        loss = torch.mean(torch.mul(-log_likelihood, target)[mask])

        return loss

class IWsoftCrossEntropy(nn.Module):
    # class_wise softCrossEntropy for class balance
    def __init__(self, ignore_index= -1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions (N, C, H, W)
        :param target: target distribution (N, C, H, W)
        :return: loss with image-wise weighting factor
        """
        assert inputs.size() == target.size()
        mask = (target != self.ignore_index)
        _, argpred = torch.max(inputs, 1)
        weights = []
        batch_size = inputs.size(0)
        for i in range(batch_size):
            hist = torch.histc(argpred[i].cpu().data.float(), 
                            bins=self.num_class, min=0,
                            max=self.num_class-1).float()
            weight = (1/torch.max(torch.pow(hist, self.ratio)*torch.pow(hist.sum(), 1-self.ratio), torch.ones(1))).to(argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)

        log_likelihood = F.log_softmax(inputs, dim=1)
        loss = torch.sum((torch.mul(-log_likelihood, target)*weights)[mask]) / (batch_size*self.num_class)
        return loss

class IW_MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio
    
    def forward(self, pred, prob, label=None):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :param label(optional): the map for counting label numbers (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        # prob -= 0.5
        N, C, H, W = prob.size()
        mask = (prob != self.ignore_index)
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = (maxpred != self.ignore_index)
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long)*self.ignore_index)
        if label is None:
            label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(), 
                            bins=self.num_class+1, min=-1,
                            max=self.num_class-1).float()
            hist = hist[1:]
            weight = (1/torch.max(torch.pow(hist, self.ratio)*torch.pow(hist.sum(), 1-self.ratio), torch.ones(1))).to(argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2,3), True).detach()
        loss = -torch.sum((torch.pow(prob, 2)*weights)[mask]) / (batch_size*self.num_class)
        return loss

class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1, num_class=19):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class

    def forward(self, pred, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        mask = (prob != self.ignore_index)    
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss


class PixelWiseCausalityLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def get_entropy_at_each_position(self, p_softmax):
        # p_softmax at spatial location (i,j)
        # exploit ENTropy minimization (ENT) to help DA,
        mask = p_softmax.ge(0.000001)
        mask_out = torch.masked_select(p_softmax, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return (entropy / float(p_softmax.size(0)))

    def get_causality_loss(self, x_IN_entropy, x_useful_entropy, x_useless_entropy):
        self.ranking_loss = torch.nn.SoftMarginLoss()
        y = torch.ones_like(x_IN_entropy)
        return self.ranking_loss(x_IN_entropy - x_useful_entropy, y) + self.ranking_loss(
            x_useless_entropy - x_IN_entropy, y)

    def forward(self, x_IN_entropy, x_useful_entropy, x_useless_entropy):
        return self.get_causality_loss(self.get_entropy(x_IN_entropy), self.get_entropy(x_useful_entropy),
                                   self.get_entropy(x_useless_entropy))
