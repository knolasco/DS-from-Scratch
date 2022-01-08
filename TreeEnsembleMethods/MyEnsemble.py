from __future__ import division
from enum import EnumMeta
import numpy as np
from MyDecisionTrees import DecisionTreeRegressor

# bagging regressor

class Bagger:
    def fit(self, X_train, y_train, B, max_depth = 100, min_size = 2, seed = None):
        
        # initialize attributes
        self.X_train = X_train
        self.N, self.D = self.X.shape
        self.y_train = y_train
        self.B = B
        self.seed = seed
        self.trees = []

        # set seed
        np.random.seed(self.seed)
        
        # loop through number of bootstraps
        for b in range(self.B):
            # make the boostrap set
            sample = np.random.choice(np.arange(self.N), size = self.N, replace = True)
            X_train_b = self.X_train[sample]
            y_train_b = self.y_train[sample]

            tree = DecisionTreeRegressor()
            tree.fit(X_train_b, y_train_b, self.max_depth, self.min_size)
            self.trees.append(tree)
    
    def predict(self, X_test):
        
        y_hat_tests = np.empty((len(self.trees), len(X_test)))
        for i, tree in enumerate(self.trees):
            y_hat_tests[i] = tree.predict(X_test)

        return y_hat_tests.mean(0)
