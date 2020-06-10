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
	weight_avg = []
	rec_avg = []
	rec_period = []
	weight_period = []
	with open(fileName,"r") as f:
		for num,line in enumerate(f):
			if num == 0:
				continue
			arr = line.split()
			global_step = arr[0].replace('[',"").replace(']',"")
			step = int(global_step)
			if num == 1:
				step = 1
			## KL loss
			rec_loss = float(arr[1].split(':')[1])
			weight = float(arr[-1].split(':')[1])
			weight_period.append(weight)
			rec_period.append(rec_loss)
			# ## average result
			if (num) % period == 0 or num+1 >= max_num:
				steps.append(step)
				rec_avg.append(np.mean(rec_period))
				weight_avg.append(np.mean(weight_period))
				weight_period = []
				rec_period = []
			if num+1 >= max_num:
				break

	# print(steps, KL_avg, rec_avg)
	return steps, weight_avg, rec_avg
	

'''
Fun: plot figure
'''
def plot_figure(x, y, label_lst, x_title, location, fig_name, y_name):
	# fig = plt.figure()
	fig, ax = plt.subplots()
	# axes= plt.axes()
	linewidth = 2.2 #linewidth
	colors = config.colors
	# colors = ['blue', 'black','red','orange','darkgreen','fuchsia','blue','grey','pink','grey','coral']
	markers = ['', '','','', '', '', '', '',' ^','v','d','+']
	linestyles = ['-','-', '--','--', '--', '--','--','--']*2
	edgecolors = ['#1B2ACC','#CC4F1B','#3F7F4C']
	facecolors = ['#089FFF', '#FF9848', '#7EFF99']
	n = len(y)
	print("# of y:",n)
	for i in range(n):
		# print(y[i][0])
		plt.plot(x, y[i][0], marker = markers[i], color = colors[i], linestyle=linestyles[i],\
			lw = linewidth, markersize=5, label = label_lst[i])
		error = y[i][1]
		# print('error', error)
		plt.fill_between(x, y[i][0]-error, y[i][0]+error, \
						alpha=0.5, edgecolor=edgecolors[i], facecolor=facecolors[i],linewidth=1)
		
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
	if y_name == 'Reconstruction Loss':
		plt.ylim(10, 150)
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
	max_num = 60000
	path_list = config.path_list

	## for file name
	x_steps = []
	rec_list = []
	weights_list = []
	for path in path_list:
		weight_each = []
		rec_each = []
		for i in range(1,6):
			path_folder = path + str(i)
			fileName = os.path.join(path_folder, 'train.log')
			steps, weight_avg, rec_avg = _read_file(fileName, max_num, period)
			x_steps = steps
			if 'gamma4' in path:
				weight_avg = len(steps) * [4]
			elif 'gamma100' in path:
				weight_avg = len(steps) * [100]
			weight_each.append(weight_avg)
			rec_each.append(rec_avg)
		## compute the avg
		weights_mean = np.mean(weight_each, axis=0)
		weight_var = np.std(weight_each, axis=0)
		## recon loss
		rec_mean = np.mean(rec_each, axis=0)
		rec_var = np.std(rec_each, axis=0)
		weights_list.append([weights_mean, weight_var])
		rec_list.append([rec_mean, rec_var])
		# print(weights_mean.shape)
		
	
	## plot figure with shaded area
	location = 'best'
	x_title = 'Training steps'
	label_lst = config.label_lst
	## rec loss
	fig_name = os.path.join(folderName,'Sprites_rec_loss.eps')
	y_name = 'Reconstruction Loss'
	plot_figure(x_steps, rec_list, label_lst, x_title, location, fig_name,y_name)
	## KL loss
	fig_name = os.path.join(folderName,'Sprites_weight.pdf')
	y_name = 'Weight'
	plot_figure(x_steps, weights_list, label_lst, x_title, location, fig_name, y_name)
	



if __name__ == '__main__':
	main()
	


