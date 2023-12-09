#coding:utf-8
import scipy.io as sio  
import os
import numpy as np

def data_generate(data_name, link_num, graph_num, train_flag): 
	dirs = './data/%s/%s' % (train_flag, data_name)
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	#generate data/data.txt
	data_path = './data/%s/%s/%s.txt' % (train_flag, data_name, data_name)
	if os.path.exists(data_path):
		os.remove(data_path)
	data_file = open(data_path, mode='a+')
	data_file.writelines([str(graph_num),'\n']) #1st line: `N` number of graphs;
	#for each block of text:
	for i in range(graph_num):  
		#a line contains `n l`
		#`n` is number of nodes in the current graph, and `l` is the graph indices (starting from 0)
		data_file.writelines([str(link_num),'\t',str(i),'\n'])  
		#for each block of text:
		for j in range(link_num): 
			# `t` is the indices of current node (starting from 0), and `m` is the number of neighbors of current node
			data_file.writelines([str(j),'\t',str(link_num-1)])
			#following `m` numbers indicate the neighbor indices (starting from 0)
			for k in range(link_num):
				if k!=j:
					data_file.writelines(['\t',str(k)])
			data_file.writelines(['\n'])
	data_file.close()


	#generate data/graph_label.txt
	data_path = './data/%s/%s/label.txt' % (train_flag, data_name)
	if os.path.exists(data_path):
		os.remove(data_path)
	matfn = './mat/dataset_%d_%d.mat' % (graph_num,link_num)
	data= sio.loadmat(matfn)
	channel = data['Channel']
	graph_label = data['Label']
	distance = data['Distance']
	dquan = data['Distance_quan']
	np.savetxt(data_path,graph_label)



	#1: CSI
	for i in range(graph_num):
		sub = np.transpose(channel[:,i].reshape(link_num,link_num))
		data_path = './data/%s/%s/channel_%d.txt' % (train_flag,data_name,i) 
		if os.path.exists(data_path):
			os.remove(data_path)
		np.savetxt(data_path,sub)

	##################################################################################	
	#2: distance quantization
	for i in range(graph_num):
		sub = np.transpose(dquan[:,i].reshape(link_num,link_num))
		data_path = './data/%s/%s/distance_qua_%d.txt' % (train_flag,data_name,i) 
		if os.path.exists(data_path):
			os.remove(data_path)
		np.savetxt(data_path,sub)
	##################################################################################





