import numpy as np
from math import sqrt

class normal_initialization(object):
    
    def __call__(self,x,y = None):
        if y:
            layer = np.random.uniform(-1,1,(x,y))
        else:
            layer = np.random.uniform(-1,1,(x))
            
        return layer

class Xavier_initialization(object):
    
    def __call__(self,x,y = None):
        
        if y:
            Umin,Umax = -(1/sqrt(x + y)),(1/sqrt(x + y))
            rand_layer = np.random.rand(x,y)
            layer = Umin + rand_layer * (Umax - Umin)
        else:
            Umin,Umax = -(1/sqrt(x)),(1/sqrt(x))
            rand_layer = np.random.rand(x)
            layer = Umin + rand_layer * (Umax - Umin)
            
        return layer
    
class Xavier_ReLu_initialization(object):
    
    def __call__(self,x, y = None):
        if y:
            std = sqrt(2/x)
            rand_layer = np.random.randn(x,y)
            layer = rand_layer * std
        else:
            std = sqrt(2/x)
            rand_layer = np.random.randn(x)
            layer = rand_layer * std
        return layer
    
