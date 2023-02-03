import numpy as np
from dataclasses import dataclass,field
from ._eval import training_model


@dataclass
class layers_propperty:
    layer_id: int = field()
    dense: int = field()
    initializer: object = field()
    activation: object = field()
    weight_layer: np.ndarray = field( default= None )
    bias_layer: np.ndarray = field( default=None )
    dimension: tuple = field( default=None )
    forward_layer: list[np.ndarray] = field(default=None)
    backward_layer: list[np.ndarray] = field(default=None)



class model(training_model):
    
    layer_storage = list()
    
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def add_hdl_layer(self, dense,initializer,activation):
        
        dimension = tuple()
        if len(self.layer_storage) != 0:
            dimension = (self.layer_storage[-1].dense,dense)
        self.layer_storage.append(
            layers_propperty(
                layer_id = len(self.layer_storage) + 1,
                dense = dense,
                initializer = initializer,
                activation = activation,
                dimension = dimension
            )
        )
        
    @classmethod
    def add_weight(self, input_layer: int, classses: int, final_layer_activation = None):
        
        for layer in self.layer_storage:
            
            if layer.layer_id == 1:
                layer.weight_layer = layer.initializer( input_layer, layer.dense )
                layer.bias_layer = layer.initializer( layer.dense )
                layer.dimension = ( input_layer, layer.dense )
                continue
            
            layer.weight_layer = layer.initializer( layer.dimension[0], layer.dimension[1] )
            layer.bias_layer = layer.initializer( layer.dense )
        
        # add final layer
        last_layer = self.layer_storage[-1]
        self.layer_storage.append(
            layers_propperty(
                layer_id = len(self.layer_storage) + 1,
                dense = classses,
                initializer = last_layer.initializer,
                activation = last_layer.activation,
                dimension = (last_layer.dimension[1],classses),
                weight_layer = last_layer.initializer( last_layer.dense,classses ),
                bias_layer = last_layer.initializer( classses )
            )
        )