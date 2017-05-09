#!/bin/bash/env python
#-*- coding: utf-8
__author__ = 'ZhangJin'

import scipy.io as scio
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:
    def __init__(self):
        self.X = self.datasets()

    def datasets(self):
        X = []
        path = '/Users/zhangjin/Documents/graduate/mi project/mi dataset/sub3_wangdaming.mat'
        data = scio.loadmat(path)
        features = data['f']
        labels = [l[0] for l in data['eeglabel']]
        for f, l in zip(features, labels):
            if l == 1:
                X.append((f, 1))
            else:
                X.append((f, -1))
        return X
'''
    def nbla(self):
        X = self.X
        f11 = [item[0][0] for item in X if item[1] == 1]  # 第一类的第一个特征
        f12 = [item[0][0] for item in X if item[1] == -1]  # 第二类的第一个特征
        f21 = [item[0][1] for item in X if item[1] == 1]  # 第一类的第二个特征
        f22 = [item[0][1] for item in X if item[1] == -1]  # 第二类的第二个特征

        q = 11
        hist11, bins11 = np.histogram(f11, bins=[min(f11) + i * ((max(f11) - min(f11)) / q) for i in range(q + 1)])
        hist12, bins12 = np.histogram(f12, bins=[min(f12) + i * ((max(f12) - min(f12)) / q) for i in range(q + 1)])
        hist21, bins21 = np.histogram(f21, bins=[min(f21) + i * ((max(f21) - min(f21)) / q) for i in range(q + 1)])
        hist22, bins22 = np.histogram(f22, bins=[min(f22) + i * ((max(f22) - min(f22)) / q) for i in range(q + 1)])

        f1c1 = [h / float(30) for h in hist11]
        f1c2 = [h / float(30) for h in hist12]
        f2c1 = [h / float(30) for h in hist21]
        f2c2 = [h / float(30) for h in hist22]
        return bins11,bins12,bins21,bins22,f1c1,f1c2,f2c1,f2c2

    def nbpredict(self,x):
        bins11, bins12, bins21, bins22,f1c1, f1c2, f2c1, f2c2 = self.nbla()
        for idx, t in enumerate(bins11):
            if x[0] <= t:
                p11 = f1c1[idx-1]
                break
        for idx, t in enumerate(bins12):
            if x[0] <= t:
                p12 = f1c2[idx-1]
                break
        for idx, t in enumerate(bins21):
            if x[1] <= t:
                p21 = f2c1[idx-1]
                break
        for idx, t in enumerate(bins22):
            if x[1] <= t:
                p22 = f2c2[idx-1]
                break
        p1 = p11 * p21
        p2 = p12 * p22
        if p1 >= p2:
            return 1
        else:
            return -1

    def test(self):
        cnt = 0
        X = self.X
        for f,l in X:
            predict_label = self.nbpredict(f)
            if predict_label == l:
                cnt += 1
        accuracy = cnt / 60
        return accuracy
'''

gnb = GaussianNB()
nb = NaiveBayes()
X = nb.datasets()
f = [item[0] for item in X]
l = [item[1] for item in X]
y_pred = gnb.fit(f, l).predict(f)
print("Number of mislabeled points out of a total %d points : %d"
      % (len(f),(l != y_pred).sum()))
