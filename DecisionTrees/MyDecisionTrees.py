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

        # initialize attributes
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape
        