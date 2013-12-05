'''
Created on 2013/06/27

@author: tokichie
'''

import numpy as np
import random as rnd
#import math

class neuralnet(object):

    '''
    ' constructor
    ' @param inp: the number of inputs
    ' @param hdn: the number of hidden units
    ' @param out: the number of outputs 
    ' @param eta: learning rate
    '''
    def __init__(self, inp, hdn, out, eta, initval):
        self.inp_out = np.ones(shape=(inp+1, 1))
        self.hdn_out = np.ones(shape=(hdn+1, 1))
        self.out_out = np.zeros(shape=(out, 1))
#        self.w_inp2hdn = \
#            np.array([[0.] + [rnd.uniform(-.3, .3) for _ in range(inp)] for _ in range(hdn)])
#        self.w_hdn2out = \
#            np.array([[0.] + [rnd.uniform(-.3, .3) for _ in range(hdn)] for _ in range(out)])
        self.w_inp2hdn = \
            np.array([[0.] + [rnd.uniform(-initval, initval) for _ in range(inp)] for _ in range(hdn)])
#            np.array([[0.] + [initval for _ in range(inp)] for _ in range(hdn)])
        self.w_hdn2out = \
            np.array([[0.] + [rnd.uniform(-initval, initval) for _ in range(hdn)] for _ in range(out)])
#            np.array([[0.] + [initval for _ in range(hdn)] for _ in range(out)])
        self.learningrate = eta
        self.error = 0.0
        self.params = None
        
    def learn(self, inp, teacher, maxerr):
        cn = 0
        while (True) :
            self.error = 0.0
            for n, i in enumerate(inp):
                self.forwardprop(i)
                self.backprop(i, teacher[n])
                if (self.error <= maxerr) :
                    print self.error, i, self.out_out
                    return
                cn += 1
            print cn, self.error
            if (cn > 30000):
                return
            
    def savecoeffs(self, filename):
        np.savez(filename, w0=self.w_inp2hdn, w1=self.w_hdn2out, \
                 param=(self.inp_out.size-1, self.hdn_out.size-1, self.out_out.size, self.learningrate))
        
    def loadcoeffs(self, filename):
        tmp = np.load(filename)
        self.w_inp2hdn = tmp['w0']
        self.w_hdn2out = tmp['w1']
        self.params = tmp['param']

    def printcoeefs(self):
        print self.params
        
    def forwardprop(self, inp):
        self.inp_out[1:] = np.array([inp]).T

        hdn_act = np.dot(self.w_inp2hdn, self.inp_out)   
        #self.hdn_out[1:] = 1. / (1. + np.exp(-self.hdn_act)) #logistic sigmoid
        self.hdn_out[1:] = np.tanh(hdn_act) #sigmoid
        
        out_act = np.dot(self.w_hdn2out, self.hdn_out)
        #self.out_out = out_act    # 1 class
        self.out_out = np.exp(out_act) / np.sum(np.exp(out_act))

        
    def backprop(self, inp, teacher):
        delta_o = self.out_out - np.array([teacher]).T
        self.error += np.sum(delta_o ** 2) * 0.5      
        self.w_hdn2out -= self.learningrate * np.dot(delta_o, self.hdn_out.T)
        
        delta_h = np.dot(self.w_hdn2out.T, delta_o)
        delta_h *= 1. - self.hdn_out**2
        self.w_inp2hdn -= self.learningrate * np.dot(delta_h[1:], self.inp_out.T)

        
                