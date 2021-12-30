from __future__ import division
import numpy as np
from numpy.lib.arraysetops import unique

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
        for _, k in enumerate(self.unique_y):
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
    

    # prepare for classifications
    def _mvn_density(self, x_n, mu_k, Sigma):
        x_n_minus_mu_k = (x_n - mu_k)
        density = np.exp(-(1/2)*x_n_minus_mu_k.T @ np.linalg.inv(Sigma) @ x_n_minus_mu_k)
        return density
    
    def classify(self, X_test):
        y_n = np.empty(len(X_test))
        for i, x_n in enumerate(X_test):
            x_n = x_n.reshape(-1,1)
            p_ks = np.empty(len(self.unique_y))

            for j, _ in enumerate(self.unique_y):
                p_x_given_y = self._mvn_density(x_n, self.mu_ks[j], self.Sigma)
                p_y_given_x = self.pi_ks[j]*p_x_given_y
                p_ks[j] = p_y_given_x
        
            y_n[i] = self.unique_y[np.argmax(p_ks)]
        return y_n

class QDA:

    """
    QDA is implemented very similarly to LDA, except each observation has it's own Sigma (covariance matrix)
    """

    def fit(self, X, y):

        # initialize attributes
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape

        # save prior probabilities
        self.unique_y, unique_y_counts = np.unique(self.y, return_counts = True)
        self.pi_ks = unique_y_counts / self.N

        # calculate mu and sigma per class
        self.mu_ks = []
        self.Sigma_ks = []
        for i, k in enumerate(self.unique_y):
            X_k = self.X[self.y == k]
            mu_k = X_k.mean(0).reshape(self.D, 1)
            self.mu_ks.append(mu_k)

            Sigma_k = np.zeros((self.D, self.D))
            for x_n in X_k:
                x_n = x_n.reshape(-1,1)
                x_n_minus_mu_k = (x_n - mu_k)
                Sigma_k += np.dot(x_n_minus_mu_k, x_n_minus_mu_k.T)
            self.Sigma_ks.append(Sigma_k / len(X_k))

    # methods for the classifications
    def _mvn_density(self, x_n, mu_k, Sigma_k):
        x_n_minus_mu_k = (x_n - mu_k)
        return np.linalg.det(Sigma_k)**(-1/2)*np.exp(-(1/2)*x_n_minus_mu_k.T @ np.linalg.inv(Sigma_k) @ x_n_minus_mu_k)

    def classify(self, X_test):
        y_n = np.empty(len(X_test))
        for i, x_n in enumerate(X_test):
            x_n = x_n.reshape(-1,1)
            p_ks = np.empty(len(self.unique_y))

            for j, _ in enumerate(self.unique_y):
                p_x_given_y = self._mvn_density(x_n, self.mu_ks[j], self.Sigma_ks[j])
                p_y_given_x = self.pi_ks[j]*p_x_given_y
                p_ks[j] = p_y_given_x

            y_n[i] = self.unique_y[np.argmax(p_ks)]

        return y_n