from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# define helper functions for BinaryLogisticRegression
def logistic(z):
    return (1 + np.exp(-z))**(-1)

def standard_scaler(X):
    mean = X.mean(0)
    sd = X.std(0)
    return (X - mean)/sd

class BinaryLogisticRegression:

    def fit(self, X, y, n_iter, lr, standardize = True, has_intercept = True):

        # standardize and intercept
        if standardize:
            X = standard_scaler(X)
        if not has_intercept:
            ones = np.ones(X.shape[0]).reshape(-1,1)
            X = np.concatenate((ones, X), axis = 1)
        
        # initialize attributes
        self.X = X
        self.N, self.D = X.shape
        self.y = y
        self.n_iter = n_iter
        self.lr = lr

        # calculate beta
        beta = np.random.randn(self.D)
        for i in range(n_iter):
            p = logistic(np.dot(self.X, beta)) # vector of prorbabilities
            gradient = -np.dot(self.X.T, (self.y - p)) # finding the gradient
            beta -= self.lr*gradient # updating beta in the gradient's direction
        
        # return values
        self.beta = beta
        self.p = logistic(np.dot(self.X, self.beta))
        self.yhat = self.p.round()
    
# define helper functions for MulticlassLogisticRegression

def softmax(z):
    """
    This function is not used in the script, but shows how softmax works
    """
    return np.exp(z)/ (np.exp(z).sum())

def softmax_byrow(z):
    return (np.exp(z)/(np.exp(z).sum(1)[:, None]))

def make_I_matrix(y):
    I = np.zeros(shape = (len(y), len(np.unique(y))), dtype = int)
    for j, target in enumerate(np.unique(y)):
        I[:, j] = (y == target)
    return I

class MulticlassLogisticRegression:

    def fit(self, X, y, n_iter, lr, standardize = True, has_intercept = False):

        # standardize and intercept
        if standardize:
            X = standard_scaler(X)
        if not has_intercept:
            ones = np.ones(X.shape[0]).reshape(-1,1)
            X = np.concatenate((ones, X), axis = 1)
        
        # initialize attributes
        self.X = X
        self.N, self.D = self.X.shape
        self.y = y
        self.K = len(np.unique(y))
        self.n_iter = n_iter
        self.lr = lr

        # fit B
        B = np.random.randn(self.D*self.K).reshape((self.D, self.K))
        self.I = make_I_matrix(self.y)
        for i in range(n_iter):
            Z = np.dot(self.X, B)
            P = softmax_byrow(Z)
            gradient = np.dot(self.X.T, (self.I - P))
            B += lr*gradient
        
        # return values through attributes
        self.B = B
        self.Z = np.dot(self.X, self.B)
        self.P = softmax_byrow(self.Z)
        self.yhat = self.P.argmax(1)
    
# define helper functions for perceptron

def sign(a):
    """
    returns sign of a
    """
    return (-1)**(a < 0)

def to_binary(y):
    """
    returns 0 or 1
    """
    return y > 0