from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# define helper functions
def logistic(z):
    return (1 + np.exp(-z))**(-1)

def standard_scaler(X):
    mean = X.mean(0)
    sd = X.std(0)
    return (X - mean)/sd