#!/bin/bash/env python
#-*- coding: utf-8
__author__ = 'ZhangJin'

import numpy as np
import math
import scipy.io as scio
import heapq

class Knearst:
    def __init__(self,k):
        self.X = self.datasets()
        self.k = k

    def datasets(self):
        X = []
        path = '/Users/zhangjin/Documents/graduate/mi project/mi dataset/sub1_shenjienming.mat'
        data = scio.loadmat(path)
        features = data['f']
        labels = [l[0] for l in data['eeglabel']]
        for f, l in zip(features, labels):
            if l == 1:
                X.append((f, 1))
            else:
                X.append((f, -1))
        return X

    def distance(self,x1,x2):
        return math.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)

    def choose_knearst_pts(self,x):
        k_pts = []
        k = self.k
        X = self.X
        # Initialize the position and dist array
        pos = np.array(range(k))
        dist = np.array([self.distance(x,X[i][0]) for i in range(k)])
        for index,item in enumerate(X):
            d = self.distance(x,item[0])
            if d < dist.min():
                i = heapq.nlargest(1,xrange(k),dist.__getitem__)
                dist[i[0]] = d
                pos[i[0]] = index
        for p in pos:
            k_pts.append(X[p])
        return k_pts

    def vote(self, k_pts):
        s = 0
        for item in k_pts:
            s += item[1]
        return np.sign(s)

    def kca(self):
        right_cnt = 0
        for f,l in self.X:
            k_pts = k.choose_knearst_pts(f)
            if(self.vote(k_pts) == l):
                right_cnt += 1
        return right_cnt/float(len(self.X))

k = Knearst(3)
print k.kca()

