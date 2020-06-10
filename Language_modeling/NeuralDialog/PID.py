"""
Fun: PID control for KL vanishing problem
"""

from math import exp
import tensorflow as tf
def tf_print(a,name):
    op = tf.Print(a,[a],name,summarize=200)
    return op

class PIDControl():
	"""docstring for ClassName"""
	def __init__(self, exp_KL):
		"""define them out of loop"""
		self.exp_KL = exp_KL
		self.I_k1 = tf.Variable(0.0,trainable=False)
		## W_k1 record the previous time weight W value
		self.W_k1 = tf.Variable(0.0,trainable=False)
		
	def _Kp_fun(self, Err, scale=1.0):
		return 1.0/(1.0+tf.exp(scale*Err))
		
	def pid(self, KL_loss, Kp=0.01, Ki=-0.0001):
		""" increment PID algorithm
		Input: KL_loss
		return: weight for KL loss, WL
		"""
		# KL_loss=tf_print(KL_loss,'KL_loss')
		# print("KL_check", KL_loss)

		error_k = tf.stop_gradient(self.exp_KL - KL_loss)
		## comput P control
		Pk = Kp * self._Kp_fun(error_k)
		## I control accumulate error from time 0 to T
		# self.I_k1=tf_print(self.I_k1,'I_k1')
		Ik = self.I_k1 + Ki * error_k
		# Ik=tf_print(Ik,'Ik')
		## when time = k-1
		Ik = tf.cond(self.W_k1 < 0, lambda:self.I_k1, lambda:tf.cond(self.W_k1 > 1, lambda:self.I_k1, lambda:Ik))
		# Ik = tf.cond(self.W_k1 > 1, lambda:self.I_k1, lambda:Ik)
		## update k-1 accumulated error
		op1=tf.assign(self.I_k1,Ik)  ## I_k1 = Ik

		## update weight WL
		Wk = Pk + Ik
		op2= tf.assign(self.W_k1,Wk)   ## self.W_k1 = Wk
		
		## min and max value --> 0 and 1
		## if Wk > 1, Wk = 1; if Wk<0, Wk = 0
		with tf.control_dependencies([op1,op2]):
			Wk = tf.cond(Wk > 1, lambda: 1.0, lambda: tf.cond(Wk < 0, lambda: 0.0, lambda: Wk))
		
		return Wk
		





