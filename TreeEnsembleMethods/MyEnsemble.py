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

def gini_index(y, weights):
    weighted_pmk = get_weighted_pmk(y, weights)
    return np.sum(weighted_pmk*(1-weighted_pmk))

def cross_entropy(y, weights):
    weighted_pmk = get_weighted_pmk(y, weights)
    return -np.sum(weighted_pmk*np.log2(weighted_pmk))

def split_loss(child1, child2, weights1, weights2, loss = cross_entropy):
    return (len(child1)*loss(child1, weights1) + len(child2)*loss(child2, weights2))/(len(child1) + len(child2))


# adding decision tree classifier used for ADA
# slightly different than classifier in MyDecisionTree.py
# this classifier keeps track of the ID so that we could use the weights

## Helper Classes
class Node:
    
    def __init__(self, Xsub, ysub, observations, ID, depth = 0, parent_ID = None, leaf = True):
        self.Xsub = Xsub
        self.ysub = ysub
        self.observations = observations
        self.ID = ID
        self.size = len(ysub)
        self.depth = depth
        self.parent_ID = parent_ID
        self.leaf = leaf
        

class Splitter:
    
    def __init__(self):
        self.loss = np.inf
        self.no_split = True
        
    def _replace_split(self, loss, d, dtype = 'quant', t = None, L_values = None):
        self.loss = loss
        self.d = d
        self.dtype = dtype
        self.t = t
        self.L_values = L_values  
        self.no_split = False

        
## Main Class
class DecisionTreeClassifier:
    
    #############################
    ######## 1. TRAINING ########
    #############################
    
    ######### FIT ##########
    def fit(self, X, y, weights, loss_func = cross_entropy, max_depth = 100, min_size = 2, C = None):
        
        ## Add data
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape
        dtypes = [np.array(list(self.X[:,d])).dtype for d in range(self.D)]
        self.dtypes = ['quant' if (dtype == float or dtype == int) else 'cat' for dtype in dtypes]
        self.weights = weights
        
        ## Add model parameters
        self.loss_func = loss_func
        self.max_depth = max_depth
        self.min_size = min_size
        self.C = C
        
        ## Initialize nodes
        self.nodes_dict = {}
        self.current_ID = 0
        initial_node = Node(Xsub = X, ysub = y, observations = np.arange(self.N), ID = self.current_ID, parent_ID = None)
        self.nodes_dict[self.current_ID] = initial_node
        self.current_ID += 1
        
        # Build
        self._build()

    ###### BUILD TREE ######
    def _build(self):
        
        eligible_buds = self.nodes_dict 
        for layer in range(self.max_depth):
            
            ## Find eligible nodes for layer iteration
            eligible_buds = {ID:node for (ID, node) in self.nodes_dict.items() if 
                                (node.leaf == True) &
                                (node.size >= self.min_size) & 
                                (~ct.all_rows_equal(node.Xsub)) &
                                (len(np.unique(node.ysub)) > 1)}
            if len(eligible_buds) == 0:
                break
            
            ## split each eligible parent
            for ID, bud in eligible_buds.items():
                                
                ## Find split
                self._find_split(bud)
                
                ## Make split
                if not self.splitter.no_split:
                    self._make_split()
                
    ###### FIND SPLIT ######
    def _find_split(self, bud):
        
        ## Instantiate splitter
        splitter = Splitter()
        splitter.bud_ID = bud.ID
        
        ## For each (eligible) predictor...
        if self.C is None:
            eligible_predictors = np.arange(self.D)
        else:
            eligible_predictors = np.random.choice(np.arange(self.D), self.C, replace = False)
        for d in sorted(eligible_predictors):
            Xsub_d = bud.Xsub[:,d]
            dtype = self.dtypes[d]
            if len(np.unique(Xsub_d)) == 1:
                continue

            ## For each value...
            if dtype == 'quant':
                for t in np.unique(Xsub_d)[:-1]:
                    L_condition = Xsub_d <= t
                    ysub_L = bud.ysub[L_condition]
                    ysub_R = bud.ysub[~L_condition]
                    weights_L = self.weights[bud.observations][L_condition]
                    weights_R = self.weights[bud.observations][~L_condition]
                    loss = split_loss(ysub_L, ysub_R,
                                      weights_L, weights_R,
                                      loss = self.loss_func)
                    if loss < splitter.loss:
                        splitter._replace_split(loss, d, 'quant', t = t)
            else:
                for L_values in ct.possible_splits(np.unique(Xsub_d)):
                    L_condition = np.isin(Xsub_d, L_values)
                    ysub_L = bud.ysub[L_condition]
                    ysub_R = bud.ysub[~L_condition]
                    weights_L = self.weights[bud.observations][L_condition]
                    weights_R = self.weights[bud.observations][~L_condition]
                    loss = split_loss(ysub_L, ysub_R,
                                      weights_L, weights_R,
                                      loss = self.loss_func)
                    if loss < splitter.loss: 
                        splitter._replace_split(loss, d, 'cat', L_values = L_values)
                        
        ## Save splitter
        self.splitter = splitter
    
    ###### MAKE SPLIT ######
    def _make_split(self):
        
        ## Update parent node
        parent_node = self.nodes_dict[self.splitter.bud_ID]
        parent_node.leaf = False
        parent_node.child_L = self.current_ID
        parent_node.child_R = self.current_ID + 1
        parent_node.d = self.splitter.d
        parent_node.dtype = self.splitter.dtype
        parent_node.t = self.splitter.t        
        parent_node.L_values = self.splitter.L_values
        
        ## Get X and y data for children
        if parent_node.dtype == 'quant':
            L_condition = parent_node.Xsub[:,parent_node.d] <= parent_node.t
        else:
            L_condition = np.isin(parent_node.Xsub[:,parent_node.d], parent_node.L_values)
        Xchild_L = parent_node.Xsub[L_condition]
        ychild_L = parent_node.ysub[L_condition]
        child_observations_L = parent_node.observations[L_condition]
        Xchild_R = parent_node.Xsub[~L_condition]
        ychild_R = parent_node.ysub[~L_condition]
        child_observations_R = parent_node.observations[~L_condition]
        
        ## Create child nodes
        child_node_L = Node(Xchild_L, ychild_L, child_observations_L,
                            ID = self.current_ID, depth = parent_node.depth + 1,
                            parent_ID = parent_node.ID)
        child_node_R = Node(Xchild_R, ychild_R, child_observations_R,
                            ID = self.current_ID + 1, depth = parent_node.depth + 1,
                            parent_ID = parent_node.ID)
        self.nodes_dict[self.current_ID] = child_node_L
        self.nodes_dict[self.current_ID + 1] = child_node_R
        self.current_ID += 2
                
            
    #############################
    ####### 2. PREDICTING #######
    #############################
    
    ###### LEAF MODES ######
    def _get_leaf_modes(self):
        self.leaf_modes = {}
        for node_ID, node in self.nodes_dict.items():
            if node.leaf:
                values, counts = np.unique(node.ysub, return_counts=True)
                self.leaf_modes[node_ID] = values[np.argmax(counts)]
    
    ####### PREDICT ########
    def predict(self, X_test):
        
        # Calculate leaf modes
        self._get_leaf_modes()
        
        yhat = []
        for x in X_test:
            node = self.nodes_dict[0] 
            while not node.leaf:
                if node.dtype == 'quant':
                    if x[node.d] <= node.t:
                        node = self.nodes_dict[node.child_L]
                    else:
                        node = self.nodes_dict[node.child_R]
                else:
                    if x[node.d] in node.L_values:
                        node = self.nodes_dict[node.child_L]
                    else:
                        node = self.nodes_dict[node.child_R]
            yhat.append(self.leaf_modes[node.ID])
        return np.array(yhat)


# start building ADA boost

class AdaBoost:

    def fit(self, X_train, y_train, T, stub_depth = 1):
        
        # initialize attributes
        self.X_train = X_train
        self.N, self.D = self.X_train.shape
        self.y_train = y_train
        self.T = T
        self.stub_depth = stub_depth
        