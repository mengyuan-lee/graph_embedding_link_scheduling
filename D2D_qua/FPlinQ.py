#coding:utf-8
import numpy as np
import math
import random



def dB_trans(a):
	b=math.pow(10, a/10)
	return b

def object_rate_sum(H, P_dBm, x, link_num):
	P = dB_trans(P_dBm-30)
	noise_density = -169 #dBm/Hz
	bandwidth = 5e+6
	p_noise_dBm = noise_density + 10*math.log10(bandwidth)
	p_noise = dB_trans(p_noise_dBm-30)


	H = H*H
	R = np.zeros((link_num,1))

	for i in range(link_num):
		sum_all = 0
		for j in range(link_num):
			sum_all = sum_all + H[j,i]*P*x[j]
		sum_ij = sum_all - H[i,i]*P*x[i] #sum of interference
		R[i] = math.log2(1+H[i,i]*P*x[i]/(sum_ij+p_noise))
	return sum(R)


