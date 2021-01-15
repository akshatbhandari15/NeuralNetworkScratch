# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 12:11:15 2021

@author: aksha
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
sknet = MLPClassifier(hidden_layer_sizes=(32), learning_rate_init=0.001, max_iter=10000)


df = pd.read_csv("fashion-mnist_train.csv")

df.head()

labels = df.label
X = df.drop(["label"], axis = 1)
labels = labels.to_numpy()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

print(f"Shape of train set is {X.shape}")
print(f"Shape of train label is {labels.shape}")
y_label = np.zeros([60000, 10])

for i in range(len(labels)):
    y_label[i][labels[i]]= 1
    
    

Xtest = pd.read_csv("fashion-mnist_test.csv")
ytest = Xtest.label
Xtest = Xtest.drop('label', axis = 1)
Xtest = Xtest.to_numpy()

ytest_labels = np.zeros([len(ytest), 10])

for i in range(len(ytest)):
    ytest_labels[i][ytest[i]] = 1
    
    
    
    
sknet.fit(X, y_label)
preds_train = sknet.predict(X)
preds_test = sknet.predict(Xtest)

print("Train accuracy of sklearn neural network: {}".format(round(accuracy_score(preds_train, y_label),2)*100))
print("Test accuracy of sklearn neural network: {}".format(round(accuracy_score(preds_test, ytest),2)*100))
