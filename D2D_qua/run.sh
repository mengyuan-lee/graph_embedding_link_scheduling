#!/bin/bash


NODEDIM=8
EDGEDIM=8
LV=2
CONV_SIZE=32
FP_LEN=0
n_hidden1=64
n_hidden2=0
bsize=128
num_epochs=10000
learning_rate=0.01
DATA=demo
graph_num=500
link_num=50
valnum=1000
DOBN=True



python 'main.py' \
    -mode 'cpu'\
    -num_class 2\
    -data $DATA\
    -batch_size $bsize \
    -node_dim $NODEDIM\
    -edge_dim $EDGEDIM\
    -latent_dim $CONV_SIZE \
    -out_dim $FP_LEN \
    -hidden1 $n_hidden1 \
    -hidden2 $n_hidden2 \
    -max_lv $LV \
    -num_epochs $num_epochs \
    -link_num $link_num \
    -graph_num $graph_num \
    -learning_rate $learning_rate \
    -do_bn $DOBN \
    -val_num $valnum \
    $@



