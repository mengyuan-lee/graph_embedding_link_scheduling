#coding:utf-8
from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import math
import torch

from pytorch_util import weights_init

ACTIVATION = F.relu

class My_loss(nn.Module):
    def __init__(self, batch_graph, train_flag, cmd_args):
        super(My_loss, self).__init__()
        self.batch_graph = batch_graph
        self.train_flag = train_flag
        self.cmd_args = cmd_args
        
    def forward(self, prob):
        P = 10
        p_noise = 6.2946e-14
        loss =  0.0
        
        for i in range(len(self.batch_graph)):
            index = self.batch_graph[i].label
            H = np.loadtxt('./data/%s/%s/channel_%d.txt' % (self.train_flag, self.cmd_args.data,index))
            H = H*H
            R = torch.zeros((self.cmd_args.link_num,1))
            y = prob[self.cmd_args.link_num*i:self.cmd_args.link_num*(i+1),:]
            for k in range(self.cmd_args.link_num):
                sum_all = 0
                for j in range(self.cmd_args.link_num):
                    sum_all = sum_all + H[k,j]*y[j,1]*P
                sum_ij = sum_all - H[k,k]*y[k,1]*P 
                R[k] = torch.log(1+H[k,k]*y[k,1]*P/(sum_ij+p_noise))/math.log(2) 
            loss = loss +1/sum(R) 
        return loss/len(self.batch_graph)

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)
        
        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_class, batch_normalization=False):
        super(MLPClassifier, self).__init__()

        self.hidden_size2 = hidden_size2

        self.h1_weights = nn.Linear(input_size, hidden_size1)
        if self.hidden_size2:
            self.h2_weights = nn.Linear(hidden_size1, hidden_size2)
            self.h3_weights = nn.Linear(hidden_size2, num_class)
        else:
            self.h2_weights = nn.Linear(hidden_size1, num_class)
        self.do_bn = batch_normalization

        if self.do_bn:
            self.bn1 = nn.BatchNorm1d(hidden_size1, momentum=0.5)
            if self.hidden_size2:
                self.bn2 = nn.BatchNorm1d(hidden_size2, momentum=0.5)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        if self.do_bn:
            h1 = self.bn1(h1)
        h1 = ACTIVATION(h1)


        h2 = self.h2_weights(h1)
        if self.hidden_size2:
            if self.do_bn:
                h2 = self.bn2(h2)
            h2 = ACTIVATION(h2)
            logits = self.h3_weights(h2)
        else:
            logits = h2


        if y is not None:
            loss_func = nn.CrossEntropyLoss() 
            loss = loss_func(logits, y)
            pred = logits.data.max(1, keepdim=True)[1]
 
            acc = float(pred.data.eq(y.data.view_as(pred)).cpu().sum().data)/ float(y.size()[0])
            return logits, loss, acc, pred, y
        else:
            return logits


