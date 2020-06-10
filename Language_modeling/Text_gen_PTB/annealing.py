"""
two annealing methods to mitigate KL vanishing problem
"""
# import tensorflow as tf

import numpy as np
import math

def _sigmoid(x):
	return 1.0/(1.0 + math.exp(-x))


def _cost_annealing(global_step, max_beta=1.0, prior_step=10000):
	"""method: cost annealing"""
	scale = prior_step/10.0
	if global_step < prior_step:
		beta = _sigmoid((global_step-prior_step/2.0)/scale)
	else:
		beta = max_beta
	
	return beta
	

def _cyclical_annealing(global_step, period=10):
	''' Params:
		period: every N steps as a period to change beta
		global_step: training step
	'''
	period = int(period)
	R = 0.5
	tau = 1.0*(global_step % period)/period
	if tau <= R:
		beta = tau/R
	else:
		beta = 1.0
	
	return beta
	

