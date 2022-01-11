import numpy as np

# ============ Activation Functions ============
def ReLU(h):
    return np.maximum(h, 0)

def sigmoid(h):
    return 1 / (1 + np.exp(-h))

def linear(h):
    return h

activation_function_dict = {'ReLU' : ReLU,
                            'sigmoid' : sigmoid,
                            'linear' : linear}
