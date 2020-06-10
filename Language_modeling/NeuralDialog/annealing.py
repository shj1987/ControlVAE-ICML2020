"""
two annealing methods to mitigate KL vanishing problem
"""
import tensorflow as tf


def _cost_annealing(global_step, prior_step=10000):
    """method: cost annealing"""
    scale = prior_step/10.0
    global_step_int = tf.cast(global_step, tf.int32)
    global_step_float = tf.cast(global_step, tf.float32)
    w_KL = tf.cond(global_step_int < prior_step, lambda:tf.nn.sigmoid((global_step_float-prior_step/2.0)/scale), lambda:1.0)
    
    return w_KL
    


def _cyclical_annealing(global_step, period=20000):
    ''' Params:
        period: every N steps as a period to change w_KL
        global_step: training step
    '''
    # R = tf.constant(0.5)
    R = 0.5
    global_step_int = tf.cast(global_step, tf.int32)
    ## compute tau
    tau = tf.mod(global_step_int,period)
    tau = tf.cast(tau, tf.float32)/float(period)
    ## if condition
    w_KL = tf.cond(tau <= R, lambda:tau/R, lambda:1.0)
    
    return w_KL
    
    



