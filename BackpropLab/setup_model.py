import numpy as np
from dataclasses import dataclass,field
from utils import activ_fx,initializer

@dataclass
class layers_propperty:
    layer_id: int = field()
    dense: int = field()
    initializer: object = field()
    activation: object = field()
    weight: np.ndarray = field( default= None )



class model():
    
    model_storage = list()
    
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def add_layer(self, dense,initializer,activation):
        self.model_storage.append(
            layers_propperty(
                layer_id = len(self.model_storage),
                dense = dense,
                initializer = initializer,
                activation = activation
            )
        )