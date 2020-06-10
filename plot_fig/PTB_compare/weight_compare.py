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
import importlib
from typing import Any

config: Any = importlib.import_module('config_paras')



def _read_file(fileName,max_num,period=100):
	steps = []
	weight_avg = []
	rec_avg = []
	rec_period = []
	weight_period = []
	batch_size = 32
	with open(fileName,"r") as f:
		for num,line in enumerate(f):
			arr = line.strip().split()
			global_step = arr[1].split(':')[1]
			step = int(global_step)
			## KL loss
			weight = float(arr[-1].split(':')[1])
			# print(weight)
			weight_period.append(weight)
			## average result
			if num % period == 0 or num+1 >= max_num:
				steps.append(step)
				weight_avg.append(np.mean(weight_period))
				weight_period = []
			if num+1 >= max_num:
				break

	return steps, weight_avg
	

'''
Fun: plot figure
'''
def plot_figure(x, y, label_lst, x_title, location, fig_name, y_name):
	# fig = plt.figure()
	fig, ax = plt.subplots()
	# axes= plt.axes()
	linewidth = 1.8 #linewidth
	# colors = ['blue', 'red','black','green','orchid','orange','darkblue','pink','grey','coral']
	colors = config.colors
	# colors = ['blue', 'orange','darkgreen','black','red','dimgray', 'lime',\
	# 			'fuchsia','blue','grey','pink','grey','coral']

	markers = ['', '','','', '', '', '', '',' ^','v','d','+']
	linestyles = ['-','-', '--','--', '--', '--','--','--']*2
	n = len(y)
	print("# of y:",n)
	for i in range(n):
		plt.plot(x, y[i], marker = markers[i], color = colors[i], linestyle=linestyles[i],\
			lw = linewidth, markersize=5, label = label_lst[i])
	
	font2 = {'family' : 'Times New Roman','weight': 'normal','size': 14}
	plt.tick_params(labelsize = 14)
	plt.xlabel(x_title, fontsize = 14)  #we can use font 2
	plt.ylabel(y_name, fontsize = 14)
	
	# plt.xticks(x, x)#show the X values
	# plt.xticks(np.arange(0, x[-1], 10000))
	ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x/1000)) + 'K'))
	### loc = "best",'upper left' = 2,'lower left'=3
	plt.legend(loc = 'best', prop={'size': 11})
	# plt.title('Expected fusion error',fontsize = 14)
	plt.grid()
	plt.tight_layout()
	x_title = x_title.split()
	fig.savefig(fig_name,dpi = 600)
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
	max_num = 80000
	path_list = config.path_list

	# path_list = ['pid_ptb_KL1.0','pid_ptb_KL3.0',\
	# 			'cost_anneal_b32_ptb_step10000.0', 'cost_anneal_b32_ptb_step20000.0',\
	# 			'cyclical_b32_ptb_cyc_4.0', 'cyclical_b32_ptb_cyc_8.0']

	# path_list = ['cost_anneal_b32_ptb_step10000.0']
	## for file name
	x_steps = []
	weight_list = []
	for path in path_list:
		fileName = os.path.join(path, 'train.log')
		steps, weight_avg = _read_file(fileName, max_num, period)
		x_steps = steps
		weight_list.append(weight_avg)
		# print(weight_avg)
		# break
	# print(rec_list)

	## plot figure
	location = 'best'
	x_title = 'Training steps'
	label_lst = config.label_lst
	# label_lst =['ControlVAE-KL-2', 'ControlVAE-KL-3',\
	# 			'KL-anneal-10000','KL-anneal-20000','cyclical-4','cyclical-8']
	## rec loss
	fig_name = os.path.join(folderName,'PTB_weight.eps')
	y_name = 'Weight'
	plot_figure(x_steps, weight_list, label_lst, x_title, location, fig_name,y_name)
    
	




if __name__ == '__main__':
	main()



