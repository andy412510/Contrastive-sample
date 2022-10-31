from __future__ import absolute_import
import os, sys
import functools
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from .mmd import *
from torch.nn import init

class GANLoss_MUNIT(nn.Module):
    def __init__(self, smooth=False):
        super(GANLoss_MUNIT, self).__init__()
        self.smooth = smooth

    def __call__(self, input, target_is_real):  # run here
        real_label = 1.0
        fake_label = 0.0
        if self.smooth:
            real_label = random.uniform(0.7,1.0)
            fake_label = random.uniform(0.0,0.3)
        loss = 0
        for it, pred in enumerate(input):
            # data = F.sigmoid(pred)
            data = pred
            if target_is_real:
                loss += torch.mean((data - real_label)**2)
            else:
                loss += torch.mean((data - fake_label)**2)
        return loss


class GANLoss(nn.Module):
    def __init__(self, smooth=False):
        super(GANLoss, self).__init__()
        self.smooth = smooth

    def __call__(self, input, target_is_real):  # run here
        real_label = 1.0
        fake_label = 0.0
        if self.smooth:
            real_label = random.uniform(0.7,1.0)
            fake_label = random.uniform(0.0,0.3)
        loss = 0
        gan_loss = nn.MSELoss()
        for it, pred in enumerate(input):
            # data = F.sigmoid(pred)
            data = pred
            if target_is_real:
                target_tensor = torch.ones_like(data).fill_(real_label)
            else:
                target_tensor = torch.zeros_like(data).fill_(fake_label)
            loss += gan_loss(data, target_tensor)
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap, dim=0)
        dist_an = torch.stack(dist_an, dim=0)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec

class MMDLoss(nn.Module):
    def __init__(self,
                 base=1.0,
                 sigma_list=[1, 2, 10]):
        super(MMDLoss, self).__init__()

        # sigma for MMD
        self.base = base
        self.sigma_list = sigma_list
        self.sigma_list = [sigma / self.base for sigma in self.sigma_list]

    def forward(self, Target, Source):
        """
            Args:
                predict: batch size x 2048 x 1 x 1 -> batch size x 2048
                gt:      batch size x 1
        """

        Target = Target.view(Target.size()[0], -1)
        Source = Source.view(Source.size()[0], -1)

        mmd2_D = mix_rbf_mmd2(Target, Source, self.sigma_list)
        mmd2_D = F.relu(mmd2_D)
        mmd2_D = torch.sqrt(mmd2_D)

        return mmd2_D