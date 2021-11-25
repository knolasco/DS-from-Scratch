import numpy as np
from __future__ import division

# define scaler function like the scikit-learn scaler
def standard_scaler(X):
    means = X.mean(0)
    stds = X.std(0)
    return (X - means)/stds

