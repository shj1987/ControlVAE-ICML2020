""" 
Fun: compare the PID with other annealing method
KL vanishing comparison
Weight increas
Reconstruction error
"""

import os
import csv,json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import math
import importlib
from typing import Any

config: Any = importlib.import_module('config_paras')


def _read_file(fileName,max_num,period=100):
	steps = []
	KL_avg = []
	KL_period = []
	total_KL_avg = []
	total_kl_period = []
	step = 0
	with open(fileName,"r") as f:
		for num,line in enumerate(f):
			arr = line.split()
			step += 20
			## KL loss
			total_kl = float(arr[0].split(':')[1])
			total_kl_period.append(total_kl)
			## wise element
			kl_loss = arr[1].split(':')[1]
			wise_KL = kl_loss.split(',')
			wise_KL = [float(k) for k in wise_KL]
			KL_period.append(wise_KL)
			# ## average result
			if (num) % period == 0 or num+1 >= max_num:
				steps.append(step)
				mean_total = np.mean(total_kl_period)
				mean_wise_kl = np.mean(KL_period,axis=0)
				# print(np.append(mean_wise_kl,mean_total))
				KL_avg.append(np.append(mean_wise_kl, mean_total))
				KL_period = []
				total_kl_period = []
			if num+1 >= max_num:
				break

	steps[0] = 1
	return steps, KL_avg
	

'''
Fun: plot figure
'''
def plot_figure(x, y, label_lst, x_title, location, fig_name, y_name):
	# fig = plt.figure()
	fig, ax = plt.subplots()
	# axes= plt.axes()
	linewidth = 2.5 #linewidth
	colors = config.colors
	# colors = ['blue', 'black','red','orange','darkgreen','fuchsia','blue','grey','pink','grey','coral']
	markers = ['', '','','', '', '', '', '']*4
	linestyles = ['-','-', '-','-', '-', '-','-','-']*5
	# edgecolors = ['#1B2ACC','#CC4F1B','#3F7F4C']
	# facecolors = ['#089FFF', '#FF9848', '#7EFF99']
	n = len(y)
	print("# of y:",n)
	for i in range(n):
		# print(y[i][0])
		plt.plot(x, y[i], marker = markers[i], color = colors[i], linestyle=linestyles[i],\
			lw = linewidth, markersize=5, label = label_lst[i])
	
	font2 = {'family' : 'Times New Roman','weight': 'normal','size': 14}
	plt.tick_params(labelsize = 15)
	plt.xlabel(x_title, fontsize = 15)  #we can use font 2
	plt.ylabel(y_name, fontsize = 15)
	
	# plt.xticks(x, x)#show the X values
	# plt.xticks(np.arange(0, x[-1], 10000))
	ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x/1000)) + 'K'))
	### loc = "best",'upper left' = 2,'lower left'=3
	plt.legend(loc = 'best', prop={'size': 10.5})
	stepsize = 3
	# start, end = ax.get_xlim()
	ax.yaxis.set_ticks(np.arange(0, 20, stepsize))
	# plt.ylim(0, 19)
	# plt.title('Expected fusion error',fontsize = 14)
	plt.grid()
	plt.tight_layout()
	x_title = x_title.split()
	fig.savefig(fig_name,bbox_inches='tight',dpi = 600)
	plt.show()
	

def _create_folder(folderName):
	if not os.path.exists(folderName):
		os.makedirs(folderName)


## main function
def main():
	## compare the hit ratio
	folderName = 'figures'
	_create_folder(folderName)

	period = 100
	max_num = 50000
	path = 'dsprites_PID_c18-v3'
	## for file name
	fileName = os.path.join(path, 'train.kl')
	steps, KL_avg = _read_file(fileName, max_num, period)
	x_steps = steps
	
	## plot figure with shaded area
	location = 'best'
	x_title = 'Training steps'
	## look at the gif to find the factors
	label_lst = ['z1','z2 (y)','z3 (Scale)','z4 (Shape)','z5','z6 (x)','z7 (Orientation)',\
				'z8','z9','z10','total KL']
	# ## rec loss
	KL_trans = np.transpose(KL_avg)
	# print(KL_trans[6:11,:100000])
	fig_name = os.path.join(folderName,'Sprites_KL_loss.eps')
	y_name = 'KL Divergence'
	plot_figure(x_steps, KL_trans, label_lst, x_title, location, fig_name,y_name)
	


if __name__ == '__main__':
	main()
	


