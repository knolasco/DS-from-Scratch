from __future__ import division
import numpy as np

class LDA:
    
    # fit the model
    def fit(self, X, y):

        # initialize the attributes
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape

        # save prior distributions
        self.unique_y, unique_y_counts = np.unique(self.y, return_counts = True)
        self.pi_ks = unique_y_counts / self.N

        # get the mu for each class
        self.mu_ks = []
        self.Sigma = np.zeros((self.D, self.D))
        for i, k in enumerate(self.unique_y):
            X_k = self.X[self.y == k] # get all X for class k
            mu_k = X_k.mean(0).reshape(self.D,1)
            self.mu_ks.append(mu_k)

            # calculate overall sigma
            for x_n in X_k:
                x_n = x_n.reshape(-1, 1)
                x_n_minus_mu_k = (x_n - mu_k)
                self.Sigma += np.dot(x_n_minus_mu_k, x_n_minus_mu_k.T)
    
        # save final sigma
        self.Sigma /= self.N