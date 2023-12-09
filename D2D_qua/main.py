#coding:utf-8
import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_generate as dg
import FPlinQ as FP


sys.path.append('%s/../s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField
from mlp import MLPClassifier

from util import cmd_args, load_data

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        self.s2v = model(latent_dim=cmd_args.latent_dim, 
                        output_dim=cmd_args.out_dim,
                        num_node_feats=cmd_args.node_dim, 
                        num_edge_feats=cmd_args.edge_dim,
                        max_lv=cmd_args.max_lv,
                        batch_normalization=cmd_args.do_bn)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size1=cmd_args.hidden1, hidden_size2=cmd_args.hidden2, num_class=cmd_args.num_class, batch_normalization=cmd_args.do_bn)

    def PrepareFeatureLabel(self, batch_graph, train_flag):
        labels_np = np.zeros((cmd_args.link_num, len(batch_graph)))
        n_nodes = 0
        n_edges = 0

        #raad label for each graph
        labels_total = np.loadtxt('./data/%s/%s/label.txt' % (train_flag, cmd_args.data))


        for i in range(len(batch_graph)):
            index = batch_graph[i].label
            labels_np[:,i] = labels_total[:,index]
            n_nodes += batch_graph[i].num_nodes 
            n_edges += batch_graph[i].num_edges 

        labels = torch.from_numpy(labels_np.transpose().reshape(-1)).long()


        #read features for each node
        if cmd_args.node_dim > 0:
            node_i = 0
            node_feat_np = np.zeros((n_nodes,1))
            for i in range(len(batch_graph)):
                index = batch_graph[i].label
                csi = np.loadtxt('./data/%s/%s/distance_qua_%d.txt' % (train_flag, cmd_args.data, index))
                for j in range(batch_graph[i].num_nodes):
                    node_feat_np[node_i,:] = csi[j,j]  
                    node_i = node_i + 1

            node_contact = torch.from_numpy(node_feat_np).long()
            node_feat = torch.zeros(n_nodes, cmd_args.node_dim).scatter_(1, node_contact, 1)
        else:
            node_feat = None

        #read features for each edge
        if cmd_args.edge_dim > 0:
            edge_i = 0
            edge_feat_np = np.zeros((2*n_edges, 1))
            for i in range(len(batch_graph)):
                index = batch_graph[i].label
                edge_pairs = batch_graph[i].edge_pairs

                csi = np.loadtxt('./data/%s/%s/distance_qua_%d.txt' % (train_flag, cmd_args.data, index))
                j = 0
                while j < len(edge_pairs):
                    start = edge_pairs[j]
                    end = edge_pairs[j+1]
                    edge_feat_np[edge_i,:] = csi[start,end]
                    edge_feat_np[edge_i+1,:] = csi[end,start]  
                    edge_i = edge_i + 2
                    j = j + 2
            edge_contact = torch.from_numpy(edge_feat_np).long()
            edge_feat = torch.zeros(2*n_edges, cmd_args.edge_dim).scatter_(1, edge_contact, 1)
        else:
            edge_feat = None

        if cmd_args.mode == 'gpu':
            if node_feat is not None:
                node_feat = node_feat.cuda() 
            if edge_feat is not None:
                edge_feat = edge_feat.cuda() 
            labels = labels.cuda()
        return node_feat, edge_feat, labels

    def forward(self, batch_graph, train_flag): 
        node_feat, edge_feat, labels = self.PrepareFeatureLabel(batch_graph, train_flag)
        embed = self.s2v(batch_graph, node_feat, edge_feat)

        prediction = self.mlp(embed, labels)

        return prediction

def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size, train_flag='val'):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize  
    pbar = tqdm(range(total_iters), unit='batch') 

    n_samples = 0
    sum_fpr = 0
    sum_nnr = 0
    avg_rate = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize] 
        batch_graph = [g_list[idx] for idx in selected_idx] #get a batch
        _, loss, acc, pred, y = classifier(batch_graph, train_flag)
        

        #sum rate     
        FPR = np.zeros((len(batch_graph),1))
        NNR = np.zeros((len(batch_graph),1))
        for i in range(len(batch_graph)):
            index = batch_graph[i].label
            H = np.loadtxt('./data/%s/%s/channel_%d.txt' % (train_flag, cmd_args.data,index))
            FPR[i] = FP.object_rate_sum(H, 40, y.cpu().numpy()[cmd_args.link_num*i:cmd_args.link_num*(i+1)], cmd_args.link_num)
            NNR[i] = FP.object_rate_sum(H, 40, pred.cpu().numpy()[cmd_args.link_num*i:cmd_args.link_num*(i+1)].reshape(-1), cmd_args.link_num) 
        sum_fpr = sum_fpr + sum(FPR)
        sum_nnr = sum_nnr + sum(NNR)
        
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()


        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))

        total_loss.append(np.array([loss, acc]) * len(batch_graph))

        n_samples += len(batch_graph)

        if optimizer is None:
            avg_rate = avg_rate + np.sum(pred.cpu().numpy())
        else:
            avg_rate = avg_rate + np.sum(y.cpu().numpy())
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    avg_rate = avg_rate / n_samples /cmd_args.link_num
    return avg_loss, sum_nnr/sum_fpr, avg_rate


if __name__ == '__main__':
    #random seed
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    save_dir = './data/model/%s' % (cmd_args.data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dg.data_generate(cmd_args.data,cmd_args.link_num,cmd_args.graph_num,'train')
    dg.data_generate(cmd_args.data,cmd_args.link_num,cmd_args.val_num,'val')

    train_graphs = load_data('./data/train/%s/%s.txt' % (cmd_args.data, cmd_args.data))
    val_graphs = load_data('./data/val/%s/%s.txt' % (cmd_args.data, cmd_args.data))

    print('# train: %d, # val: %d' % (len(train_graphs), len(val_graphs)))

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    
    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))

    best_acc = None
    data_path = './data/train/%s/best.txt' % (cmd_args.data)
    if os.path.exists(data_path):
        os.remove(data_path)

    stop_flag = 0

    save_dir = './data/model/%s' % (cmd_args.data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for epoch in range(cmd_args.num_epochs):
        if stop_flag<=200:
            random.shuffle(train_idxes)
            avg_loss, avg_obj, avg_rate = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer, train_flag='train')

            print('\033[92maverage training of epoch %d: loss %.5f acc %.5f obj %.5f act %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_obj, avg_rate))
        
            if epoch%1==0:
                val_loss, val_obj, val_rate = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
                print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f obj %.5f act %.5f \033[0m' % (epoch, val_loss[0], val_loss[1], val_obj, val_rate))

                if best_acc is None or val_loss[1] > best_acc:
                    best_acc = val_loss[1]
                    data_file = open(data_path, mode='a+')
                    data_file.writelines([str(epoch),'\t',str(avg_loss[1]),'\t', str(avg_obj),'\t', str(val_loss[1]),'\t', str(val_obj),'\t', str(val_rate),'\n'])
                    data_file.close()
                    stop_flag = 0

                    print('----saving to best model since this is the best valid loss so far.----')
                    torch.save(classifier.state_dict(), save_dir + '/best.pkl')
                else:
                    stop_flag = stop_flag + 1

