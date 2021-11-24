import numpy as np

class LinearRegression:

    def fit(self, X, y, intercept = False):

        # add data and dimensions
        if intercept == True:
            ones = np.ones(len(X)).reshape(len(X), 1) # make it a column
            X = np.concatenate((ones, X), axis = 1) # add the column of ones to X
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
    
        # estimate the parameters
        XtX = np.dot(self.X.T, self.X)
        XtX_inverse = np.linalg.inv(XtX)
        Xty = np.dot(self.X.T, self.y)
        self.beta_hats = np.dot(XtX_inverse, Xty)

        # make in-sample predictions
        self.y_hat = np.dot(self.X, self.beta_hats)

        # calculate the loss
        self.L = 0.5*np.sum((self.y - self.y_hat)**2)

    def predict(self, X_test, intercept = True):

        # make predictions
        self.y_test_hat = np.dot(X_test, self.beta_hats)