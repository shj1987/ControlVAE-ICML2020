#!/usr/bin/env python3
# coding: utf-8

from math import exp

class PIDControl():
    """docstring for ClassName"""
    def __init__(self):
        """define them out of loop"""
        # self.exp_KL = exp_KL
        self.I_k1 = 0.0
        self.W_k1 = 1.0
        self.e_k1 = 0.0
        self.W_min = 0.9
        
    def _Kp_fun(self, Err, scale=1):
        return 1.0/(1.0 + float(scale)*exp(Err))
        
    def pid(self, exp_KL, kl_divergence, Kp=-0.01, Ki=-0.001, Kd=0.01):
        """
        position PID algorithm
        Input: KL_loss
        return: weight for KL loss, beta
        """
        error_k = exp_KL - kl_divergence
        ## comput U as the control factor
        assert Kp < 0, 'Kp should be negative in linear PID'

        Pk = Kp * error_k + self.W_min
        
        Ik = self.I_k1 + Ki * error_k
        
        ## window up for integrator
        if self.W_k1 < self.W_min:
            Ik = self.I_k1
            
        Wk = Pk + Ik
        self.W_k1 = Wk
        self.I_k1 = Ik
        
        ## min and max value
        if Wk < self.W_min:
            Wk = self.W_min
        
        return Wk, error_k
        

