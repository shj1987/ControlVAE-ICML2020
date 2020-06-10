#!/usr/bin/env python3
# coding: utf-8

from math import exp

class PIDControl():
    """incremental PID controller"""
    def __init__(self):
        """define them out of loop"""
        # self.exp_KL = exp_KL
        # self.I_k1 = 0.0
        self.W_k1 = 1.0
        self.e_k1 = 0.0
        self.W_min = 1

    def _Kp_fun(self, Err):
        return 1.0/(1.0 + exp(Err))
        
    def pid(self, exp_KL, kl_loss, Kp=0.01, Ki=-0.001, Kd=0.01):
        """
        Incremental PID algorithm
        Input: KL_loss
        return: weight for KL divergence, beta
        """
        error_k = exp_KL - kl_loss
        ## comput U as the control factor
        dP = Kp * (self._Kp_fun(error_k) - self._Kp_fun(self.e_k1))
        dI = Ki * error_k
        dW = dP + dI
        ## update with previous W_k1
        Wk = dW + self.W_k1
        
        self.W_k1 = Wk
        self.e_k1 = error_k
        
        ## min and max value
        if Wk < self.W_min:
            Wk = self.W_min
        
        return Wk, error_k
        

