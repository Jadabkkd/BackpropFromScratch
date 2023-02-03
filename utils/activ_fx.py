import numpy as np

class activations(object):
    ReLu = lambda x, deriv : np.maximum(0,x) if not deriv else (x > 0).astype(int)
    Sigmoid =  lambda x, deriv : 1 / (1+np.exp(-x).astype(np.float64)) if not deriv else x*(1-x)
    Softmax = lambda x, deriv : (np.exp(x) / np.sum(np.exp(x),axis=1,keepdims=True))
    Tanh = lambda x, deriv : np.tanh(x).astype(np.float64) if not deriv else 1 - x**2
    LeakyReLu = lambda x, deriv : np.where(x > 0, x, x * 0.01).astype(np.float64) if not deriv else (x > 0).astype(int)
    
