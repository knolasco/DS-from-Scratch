from __future__ import division
from enum import EnumMeta
import numpy as np
from MyDecisionTrees import DecisionTreeRegressor

# bagging regressor

class Bagger:
    def fit(self, X_train, y_train, B, max_depth = 100, min_size = 2, seed = None):
        
        # initialize attributes
        self.X_train = X_train
        self.N, self.D = self.X_train.shape
        self.y_train = y_train
        self.B = B
        self.seed = seed
        self.max_depth = max_depth
        self.min_size = min_size
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

# randomforest regressor
class RandomForest:

    def fit(self, X_train, y_train, B, C, max_depth = 100, min_size = 2, seed = None):

        # initialize attributes
        self.X_train = X_train
        self.N, self.D = self.X_train.shape
        self.y_train = y_train
        self.B = B
        self.C = C
        self.seed = seed
        self.max_depth = max_depth
        self.min_size = min_size
        self.trees = []


        np.random.seed(self.seed)
        
        # make B boostraps
        for b in range(self.B):
            sample = np.random.choice(np.arange(self.N), size = self.N, replace = True)
            X_train_b = self.X_train[sample]
            y_train_b = self.y_train[sample]

            tree = DecisionTreeRegressor()
            tree.fit(X_train_b, y_train_b, max_depth = self.max_depth, min_size = self.min_size, C = self.C)
            self.trees.append(tree)
        
    def predict(self, X_test):

        y_test_hats = np.empty((len(self.trees), len(X_test)))
        for i, tree in enumerate(self.trees):
            y_test_hats[i] = tree.predict(X_test)
        
        return y_test_hats.mean(0)


# ==================================== HELPER FUNCTIONS FOR ADA ===================================

def get_weighted_pmk(y, weights):
    ks = np.unique(y)
    weighted_pmk = [sum(weights[y == k]) for k in ks]
    return (np.array(weighted_pmk) / sum(weights))