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

from s2v_lib import S2VLIB
from pytorch_util import weights_init, gnn_spmm, is_cuda_float

ACTIVATION = F.relu

class EmbedMeanField(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv = 3, batch_normalization=False):
        super(EmbedMeanField, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

        self.do_bn = batch_normalization
        self.bns = []

        self.max_lv = max_lv

        #batch normalization
        if num_node_feats > 0:
            self.bn_input_node = nn.BatchNorm1d(num_node_feats, momentum=0.5)   
            self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        if num_edge_feats > 0:
            self.bn_input_edge = nn.BatchNorm1d(num_edge_feats, momentum=0.5)   
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)

        if self.do_bn:
            self.bn_input_message = nn.BatchNorm1d(latent_dim, momentum=0.5)
            for i in range(self.max_lv):
                bn = nn.BatchNorm1d(latent_dim, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)   
                self.bns.append(bn)

        if output_dim > 0:
            self.out_params = nn.Linear(latent_dim, output_dim)
            if self.do_bn:
                self.bn_out_linear = nn.BatchNorm1d(output_dim, momentum=0.5)
                self.bn_y_potential = nn.BatchNorm1d(output_dim, momentum=0.5)
        else:
            if self.do_bn:
                self.bn_y_potential = nn.BatchNorm1d(latent_dim, momentum=0.5)

        self.conv_params = nn.Linear(latent_dim, latent_dim)
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat): 
        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list) 

        if node_feat is not None:
            if is_cuda_float(node_feat):
                n2n_sp = n2n_sp.cuda()
                e2n_sp = e2n_sp.cuda()
            node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)

        h = self.mean_field(node_feat, edge_feat, n2n_sp, e2n_sp)
        
        return h

    def mean_field(self, node_feat, edge_feat, n2n_sp, e2n_sp):
        if node_feat is not None:
            if self.do_bn:
                node_feat = self.bn_input_node(node_feat)
            input_node_linear = self.w_n2l(node_feat)
            input_message = input_node_linear

        if edge_feat is not None:
            if self.do_bn:
                edge_feat = self.bn_input_edge(edge_feat)
            input_edge_linear = self.w_e2l(edge_feat)


            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear) 
            if node_feat is not None:
                input_message += e2npool_input
            else:
                input_message = e2npool_input



        if self.do_bn:
            input_message = self.bn_input_message(input_message)
        input_potential = ACTIVATION(input_message)



        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:  
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) 
            node_linear = self.conv_params( n2npool )
            merged_linear = node_linear + input_message

            if self.do_bn:
                merged_linear = self.bns[lv](merged_linear)


            cur_message_layer = ACTIVATION(merged_linear)

           
            lv += 1
        if self.output_dim > 0:
            out_linear = self.out_params(cur_message_layer)
            if self.do_bn:
                out_linear = self.bn_out_linear(out_linear)
            reluact_fp = ACTIVATION(out_linear)

        else:
            reluact_fp = cur_message_layer
            
        
        return ACTIVATION(reluact_fp)
        
