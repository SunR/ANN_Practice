from __future__ import division
from math import *
import numpy as np
import random

step = 1e-2

N = 6 #number of input vectors
D = 2 #dimension of input vectors

data = np.empty((6, 2))  #6 rows, 2 col (b/c 2-D vector)
labels = np.empty ((6,))

#This can be done by reading an input file
data[0] = np.array([1.2, 0.7])
data[1] = np.array([-0.3, -0.5])
data[2] = np.array([3.0, 0.1])
data[3] = np.array([-0.1, -1.0])
data[4] = np.array([-1.0, 1.1])
data[5] = np.array([2.1, -3])

labels[0] = 1
labels[1] = -1
labels[2] = 1
labels[3] = -1
labels[4] = -1
labels[5] = 1

a = 1.0
b = -2.0
c = -1.0

#training loop

def evaluateTrainingAccuracy():
    numCorrect = 0
    for i in range (len(data)):
        x = data[i, 0]
        y = data[i, 1]
        trueLabel == labels[i]
        predictedLabel = 1 if a*x + b*y + c > 0 else -1
        if predictedLabel == trueLabel:
            numCorrect += 1
    #print "num correct", numCorrect
    return numCorrect / len(data) 
    

for iter in range (400):
    #pick radom data point to feed to svm
    i = random.randrange(len(data))
    x = data[i, 0]
    y = data[i, 1]
    #compute f and find correct pull (df) -- still using -1, 0, 1 for pull
    predictedLabel = a*x + b*y + c
    trueLabel = labels[i]
    df = 0.0
    #print "predicted label", predictedLabel
    #print "true label", trueLabel
    if predictedLabel <  1 and trueLabel == 1:
        print "pulling up, predicted = ", predictedLabel, "true = ", trueLabel
        df = 1.0
    if predictedLabel > -1 and trueLabel == -1:
        print "pulling down, predicted = ", predictedLabel, "true = ", trueLabel
        df = -1.0
        
    da = x * df
    dx = a * df
    db = y *df
    dy = b * df
    dc = 1.0 * df
    #print df
    #print da, db, dc
    
    #update inputs
    a = a + step*da
    b = b + step*db
    c = c + step*dc

    print a, b, c

    if iter % 25 == 0:
        print "training accuracy at iteration", iter, "is", evaluateTrainingAccuracy()


