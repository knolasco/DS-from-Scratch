import numpy as np
from __future__ import division

def standard_scaler(X):
    """
    define scaler function like the scikit-learn scaler
    """
    means = X.mean(0)
    stds = X.std(0)
    return (X - means)/stds

def sign(x, first_element_zero = False):
    """
    function that returns the sign of the element in the array
    useful for gradient in lasso regression
    """
    signs = (-1)**(x < 0) # returns -1 or 1
    if first_element_zero:
        signs[0] = 0
    return signs

class RegularizedRegression:

    def _record_info(self, X, y, lam, intercept = True, standardize = True):
        """
        handles standardization, adds an intercept to the predictors,
        and records useful values
        """
        # standardize
        if standardize:
            X = standard_scaler(X)
        
        # add intercept
        if intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)
            X = np.concatenate((ones, X), axis = 1)
        
        # record values
        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
        self.lam = lam
    
    def fit_ridge(self, X, y, lam = 0, intercept = False, standardize = True):
        
        # record the info
        self._record_info(X, y, lam, intercept, standardize)

        # estimate params
        XtX = np.dot(self.X.T, self.X)
        I_prime = np.eye(self.D)
        I_prime[0,0] = 0
        XtX_plus_lam_inverse = np.linalg.inv(XtX + self.lam*I_prime)
        Xty = np.dot(self.X.T, self.y)
        self.beta_hats = np.dot(XtX_plus_lam_inverse, Xty)

        # get fitted values
        self.y_hat = np.dot(self.X, self.beta_hats)
    
    def fit_lasso(self, X, y, lam = 0, n_iters = 2000, 
                lr = 0.0001, intercept = False, standardize = True):

        # record the info
        self._record_info(X, y, lam, intercept, standardize)

        # estimate params
        beta_hats = np.random.randn(self.D)
        for i in range(n_iters):
            # '@' is used for matrix multiplication
            dL_dbeta = -self.X.T @ (self.y - (self.X @ beta_hats)) + self.lam*sign(beta_hats, first_element_zero = True)
            beta_hats -= lr*dL_dbeta
        self.beta_hats = beta_hats

        # get the fitted values
        self.y_hat = np.dot(self.X, self.beta_hats)
    
    def predict(self, X_test, intercept = True):
        if intercept:
            ones = np.ones(len(X_test)).reshape(len(X_test), 1)
            X_test = np.concatenate((ones, X_test), axis = 1)

        self.y_test_hat = np.dot(X_test, self.beta_hats)
    