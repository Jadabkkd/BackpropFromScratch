import numpy as np
from .forward import feedforward

class training_model( feedforward, object ):
    
    def __init__(self) -> None:
        super().__init__()

    def train(self,inputs,labels):
        a = self.forward(self.layer_storage,inputs)
        print(a)