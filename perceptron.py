#!/bin/bash/env python
#-*- coding: utf-8
__author__ = 'ZhangJin'

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import random


class Perceptron:
    def __init__(self):
        self.X = self.dataset()

    def dataset(self):
        X = []
        path = '/Users/zhangjin/Documents/graduate/mi project/mi dataset/sub1_shenjienming.mat'
        data = scio.loadmat(path)
        features = data['f']
        labels = [l[0] for l in data['eeglabel']]
        for f,l in zip(features, labels):
            if l==1:
                X.append((f,1))
            else:
                X.append((f,-1))
        return X

    def data_plot(self, it, vec = None, bias = None, save = False):
        N = len(self.X)
        cols = {1: 'r', -1: 'b'}
        for f,l in self.X:
            plt.plot(f[0], f[1], cols[l] + 'o')
#        w,b = self.pla()
        xpts = np.linspace(-1.3,-0.2)
#        plt.plot(xpts, (w[0]*xpts+b)/(-w[1]), 'k-')
        if vec != None and bias != None:
            plt.plot(xpts, (vec[0]*xpts+bias)/(-vec[1]), 'g-', lw=2)
        if save == True:
            plt.title('N = %s, Iteration = %s\n' %(str(N),str(it)))
            plt.savefig('p_N%s_it%s' % (str(N), str(it)),dpi=200, bbox_inches='tight')
#            plt.show()

    def choose_mis_points(self, vec, bias):
        mis_pts = []
        for x,y in self.X:
            if (vec.T.dot(x)+bias)*y <= 0:
                mis_pts.append((x,y))
        return mis_pts[random.randrange(0,len(mis_pts))]

    def classification_error(self,vec,bias):
        mis_pts = 0
        M = len(self.X)
        for x,y in self.X:
            if (vec.T.dot(x)+bias)*y <= 0:
                mis_pts += 1
        error = mis_pts / float(M)
        return error

    def pla(self, save=False):
        w = np.zeros(2)
        b = 0
        eta = 0.35
        it = 0
        while self.classification_error(w,b) != 0:
            it += 1
            x,y = self.choose_mis_points(w,b)
            w[0] = w[0] + eta * y * x[0]
            w[1] = w[1] + eta * y * x[1]
            b = b + eta * y
            if save:
                self.data_plot(it, vec=w, bias=b, save=True)
            if it > 40:
                break
        return w,b,it


p = Perceptron()
w,b,it = p.pla()
p.data_plot(it,w,b,save=True)
error = p.classification_error(w,b)
print 1-error




