import numpy as np

def tanh(x):
    exp_pos = np.exp(x)
    exp_neg = np.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t ** 2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
