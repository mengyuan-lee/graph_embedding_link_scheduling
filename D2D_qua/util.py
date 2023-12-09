#coding:utf-8
from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import pickle as cp
import os
import networkx as nx 

import argparse 
import ast


cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification') 

cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=128, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-node_dim', type=int, default=8, help='dimension of node feature')
cmd_opt.add_argument('-edge_dim', type=int, default=8, help='dimension of edge feature')
cmd_opt.add_argument('-num_class', type=int, default=2, help='#classes')
cmd_opt.add_argument('-num_epochs', type=int, default=10000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=int, default=32, help='dimension of latent layers')
cmd_opt.add_argument('-out_dim', type=int, default=0, help='s2v output size')
cmd_opt.add_argument('-hidden1', type=int, default=64, help='dimension of regression 1')
cmd_opt.add_argument('-hidden2', type=int, default=0, help='dimension of regression 2')
cmd_opt.add_argument('-max_lv', type=int, default=2, help='max rounds of message passing')
cmd_opt.add_argument('-link_num', type=int, default=50, help='number of links')
cmd_opt.add_argument('-graph_num', type=int, default=500, help='number of training graphs')
cmd_opt.add_argument('-val_num', type=int, default=1000, help='number of validation graphs')
cmd_opt.add_argument('-learning_rate', type=float, default=0.01, help='init learning_rate')
cmd_opt.add_argument('-do_bn', type=ast.literal_eval, default=True, help='batch normalization')
cmd_opt.add_argument('-loss_weight', type=float, default=0, help='loss weight')


cmd_args, _ = cmd_opt.parse_known_args()  

print(cmd_args)

class S2VGraph(object):
    def __init__(self, g, node_tags, label):
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags #the set of node labels
        self.label = label #graph label

        x, y = zip(*g.edges())  
        self.num_edges = len(x)        
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32) #edge_pair:(node1,node2)
        self.edge_pairs[:, 0] = x 
        self.edge_pairs[:, 1] = y 
        self.edge_pairs = self.edge_pairs.flatten() 

def load_data(dirs):
    print('loading data')

    g_list = []
    label_dict = {} 
    node_dict = {} 

    with open(dirs, 'r') as f:
        # 1st line: `N` number of graphs; then the following `N` blocks describe the graphs  
        n_g = int(f.readline().strip()) #get the number of graphs
        #for each graph
        for i in range(n_g):
            row = f.readline().strip().split() 
            n, l = [int(w) for w in row] #`n` is number of nodes in the current graph, and `l` is the graph label
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped  
            g = nx.Graph() 
            node_tags = []
            n_edges = 0
            #for each node
            for j in range(n):
                g.add_node(j) #add a new node j
                row = f.readline().strip().split()
                row = [int(w) for w in row]  #`t` is the tag of current node, and `m` is the number of neighbors of current node;
                if not row[0] in node_dict:  #row[0]==t
                    mapped = len(node_dict) 
                    node_dict[row[0]] = mapped
                node_tags.append(node_dict[row[0]])
                n_edges += row[1] #row[1]==m
                for k in range(2, len(row)):
                    g.add_edge(j, int(row[k])) #following `m` numbers indicate the neighbor indices (starting from 0).
            #assert len(g.edges()) * 2 == n_edges
            assert len(g) == n
            g_list.append(S2VGraph(g, node_tags, l)) 



    for g in g_list:
        g.label = label_dict[g.label] 

    return g_list
    