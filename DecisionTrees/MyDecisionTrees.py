from __future__ import division
import numpy as np


# =========================== HELPER FUNCTIONS ==================================

def calculate_RSS(_node):
    return sum((_node - np.mean(_node))**2)

def RSS_reduction(child_L, child_R, parent):
    """
    Used for regression decision trees.
    Measures how much a split reduces a parent node's RSS
    by subtracting the sum of the child RSS values from the parent RSS.
    This is used to determine the best split for any bud.
    """
    rss_parent = calculate_RSS(parent)
    rss_child_L = calculate_RSS(child_L)
    rss_child_R = calculate_RSS(child_R)
    return rss_parent - (rss_child_L + rss_child_R)

def sort_x_by_y(x, y):
    """
    Used when the predictors are categorical.
    Returns a sorted list of the unique categories in some predictor x
    according to the mean of the corresponding target values y.
    """
    unique_xs = np.unique(x)
    y_mean_by_x = np.array([y[x == unique_x].mean() for unique_x in unique_xs])
    ordered_xs = unique_xs[np.argsort(y_mean_by_x)]
    return ordered_xs

def all_rows_equal(X):
    """
    Checks to see if all bud's rows are equal across all predictors.
    We use this to decide if the bud will be split, or if it will turn into a leaf.
    """
    return (X == X[0]).all()

# =========================== HELPER CLASSES ==================================

class Node:
    """
    Represents the nodes on a tree.
    Identifies the ID of the node, the ID of the parent node, the sample of the
    predictors and target variable, the size, depth, and whether or not it is
    a leaf.
    """

    def __init__(self, Xsub, ysub, ID, depth = 0, parent_ID = None, leaf = True):
        self.ID = ID
        self.Xsub = Xsub
        self.ysub = ysub
        self.depth = depth
        self.leaf = leaf
        self.parent_ID = parent_ID
        self.size = len(self.ysub)
    

class Splitter:
    """
    Used to identify the best split for any bud.
    Identifies the split's reductiong in RSS, the variable - d - used to make
    the split, the variable's data type, and the threshold - t - (if qualitative)
    or the set of values - L_values - corresponding to the left child node (if categorical).
    """

    def __init__(self):
        self.rss_reduction = 0
        self.no_split = True
    
    def _replace_split(self, rss_reduction, d, dtype = 'quant', t = None, L_values = None):
        self.rss_reduction = rss_reduction
        self.d = d
        self.dtype = dtype
        self.t = t
        self.L_values = L_values
        self.no_split = False


# =========================== MAIN DECISION TREE CLASS ==================================

class DecisionTreeRegressor:

    # training/fitting
    def fit(self, X, y, max_depth = 100, min_size = 2, C = None):
        """
        The C variable is a hyperparameter used for random forests.
        We include it now so that it may be used in the next section.
        """

        # Initialize attributes
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape
        # Save dtype for each dimension in X
        dtypes = [np.array(list(self.X[:,d])).dtype for d in range(self.D)]
        # Categorize dtypes
        self.dtypes = ['quant' if (dtype == float or dtype == int) else 'cat' for dtype in dtypes]

        # Save regularization parameters
        self.max_depth = max_depth
        self.min_size = min_size
        self.C = C

        # Initialize Nodes
        self.nodes_dict = {}
        self.current_ID = 0
        initial_node = Node(Xsub = X, ysub = y, ID = self.current_ID, parent_ID = None)
        self.nodes_dict[self.current_ID] = initial_node
        self.current_ID += 1

        # Build tree
        self._build()
    
    def _build(self):
        """
        Build the tree. Iterate through each layer of the tree, based on max_depth,
        splitting each eligible bud before proceeding to the next layer. The eligible buds are
        tracked through the eligible_buds dictionary. A bug is eligible if:
            1) it does not have children (is a leaf)
            2) not smaller than min_size
            3) observations are not identical across all predictors
            4) has more than one unique value of the target variable.
        
        The process contiunues until there are no more eligible buds or we reach the max_depth
        """

        eligible_buds = self.nodes_dict # when intiated, the only bud is the initial_bud
        for layer in range(self.max_depth):

            # find the eligible buds for layer iteration
            eligible_buds = {ID:node for (ID, node) in self.nodes_dict.items() if
                                (node.leaf == True) &
                                (node.size >= self.min_size) &
                                (~all_rows_equal(node.Xsub)) &
                                (len(np.unique(node.ysub)) > 1)}
                            
            if len(eligible_buds) == 0:
                break

            # make a split for each eligible parent
            for ID, bud in eligible_buds.items():

                # find the split for the bud
                self._find_split(bud)

                # make the appropriate split
                if not self.splitter.no_split: # used for random forest
                    self._make_split()
    
    def _find_split(self, bud):
        """
        Method uses the Splitter class. Loops through all predictors and all possible splits for that
        predictor to find the split that minimizes the RSS.
        If predictor is quantitative, we loop through each unique value and calculate the RSS.
        If predictor is categorical, we call sort_x_by_y() to order the categories of x to then
        calculate the RSS reduction. 
        """

        # Initiate Splitter
        splitter = Splitter()
        splitter.bud_ID = bud.ID

        # Gather the eligible predictors (Used for Random Forests)
        if self.C is None:
            eligible_predictors = np.arange(self.D)
        else:
            # D choose C
            eligible_predictors = np.random.choice(np.arange(self.D), self.C, replace = False)
        
        # Loop through eligible predictors
        for d in sorted(eligible_predictors):
            Xsub_d = bud.Xsub[:, d] # the dth column
            dtype = self.dtypes[d]
            if len(np.unique(Xsub_d)) == 1:
                continue

            # Loop through threshold values
            if dtype == 'quant':
                for t in np.unique(Xsub_d)[:-1]:
                    ysub_L = bud.ysub[Xsub_d <= t]
                    ysub_R = bud.ysub[Xsub_d > t]
                    rss_reduction = RSS_reduction(ysub_L, ysub_R, bud.ysub)

                    # check to see if a split should be made
                    if rss_reduction > splitter.rss_reduction:
                        splitter._replace_split(rss_reduction, d = d, dtype = 'quant', t = t)
            
            else:
                ordered_x = sort_x_by_y(Xsub_d, bud.ysub)
                for i in range(len(ordered_x) - 1):
                    L_values = ordered_x[: i + 1]
                    ysub_L = bud.ysub[np.isin(Xsub_d, L_values)]
                    ysub_R = bud.ysub[~np.isin(Xsub_d, L_values)]
                    rss_reduction = RSS_reduction(ysub_L, ysub_R, bud.ysub)

                    # check to see if a split should be made
                    if rss_reduction > splitter.rss_reduction:
                        splitter._replace_split(rss_reduction, d = d, dtype = 'cat', L_values = L_values)
        
        # save information of the splitter
        self.splitter = splitter
    
    def _make_split(self):
        """
        If a bud is to be split, this method updates the parent node with the split information
        that creates the children nodes. For the parent node, we save the predictor - d - that was used to make the split
        and how, and record the ID for the child nodes.
        For each child node, record the training observations passing through the node, it's ID, the parent ID, the size and depth.
        """

        # Update the parent node
        parent_node = self.nodes_dict[self.splitter.bud_ID]
        parent_node.leaf = False # since it will now be split, it is no longer a leaf
        parent_node.child_L = self.current_ID
        parent_node.child_R = self.current_ID + 1
        parent_node.d = self.splitter.d
        parent_node.dtype = self.splitter.dtype
        parent_node.t = self.splitter.t
        parent_node.L_values = self.splitter.L_values

        # Get X and y for children nodes
        if parent_node.dtype == 'quant':
            L_condition = parent_node.Xsub[:,parent_node.d] <= parent_node.t
        else:
            L_condition = np.isin(parent_node.Xsub[: ,parent_node.d], parent_node.L_values)
        
        Xchild_L = parent_node.Xsub[L_condition]
        Xchild_R = parent_node.Xsub[~L_condition]
        ychild_L = parent_node.ysub[L_condition]
        ychild_R = parent_node.ysub[~L_condition]

        # Create child nodes
        child_node_L = Node(Xchild_L, ychild_L, ID = self.current_ID, parent_ID = parent_node.ID, depth = parent_node.depth + 1)
        child_node_R = Node(Xchild_R, ychild_R, ID = self.current_ID + 1, parent_ID = parent_node.ID, depth = parent_node.depth + 1)

        self.nodes_dict[self.current_ID] = child_node_L
        self.nodes_dict[self.current_ID + 1] = child_node_R
        self.current_ID += 2

    # Predicting
    def _get_leaf_means(self):
        """
        Calculates the average of the target variable among training observations.
        """
        self.leaf_means = {}
        for node_ID, node in self.nodes_dict.items():
            if node.leaf:
                self.leaf_means[node_ID] = node.ysub.mean()
    
    def predict(self, X_test):
        """
        Run each new observation through the built tree and return a fitted value:
        the mean target variable in the leaf.
        """

        # Calculate the leaf means
        self._get_leaf_means()

        yhat = []
        for x in X_test:
            node = self.nodes_dict[0] # start at the very top
            while not node.leaf:
                if node.dtype == 'quant':
                    if x[node.d] <= node.t: # if dimension passes threshold, go to left node, else go to right
                        node = self.nodes_dict[node.child_L]
                    else:
                        node = self.nodes_dict[node.child_R]
                else:
                    if x[node.d] in node.L_values:
                        node = self.nodes_dict[node.child_L]
                    else:
                        node = self.nodes_dict[node.child_R]
            yhat.append(self.leaf_means[node.ID])
        return np.array(yhat)
        

# =========================== DecisionTreeClassifier ============================

# =========================== HELPER FUNCTIONS ==================================

def gini_index(y):
    """
    Used to calculate the loss of a single node
    """
    size = len(y)
    _, counts = np.unique(y, return_counts = True)
    pmk = counts / size
    return np.sum(pmk*(1-pmk))

def cross_entropy(y):
    """
    Used to calculate the loss of a single node
    """
    size = len(y)
    _, counts = np.unique(y, return_counts = True)
    pmk = counts / size
    return -np.sum(pmk*np.log2(pmk))