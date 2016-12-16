import numpy as np
from numpy import genfromtxt

def getDataSet():
    dataset = genfromtxt('features.csv', delimiter='_')
    y = dataset[:, 0]
    X = dataset[:, 1:]

    dataset = genfromtxt('features -t.csv', delimiter='_')
    y_te = dataset[:, 0]
    X_te = dataset[:, 1:]

    return X, y, X_te, y_te