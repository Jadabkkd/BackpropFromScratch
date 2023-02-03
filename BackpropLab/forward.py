import numpy as np

class feedforward(object):
    
    def __init__(self) -> None:
         pass
    
    def forward(self, layers, batch_data):
            
            layer_result = list()
            
            for layer in layers:
                if layer.layer_id == 1:
                    layer_result.append(
                        layer.activation(
                            np.dot(
                                    batch_data, layer.weight_layer
                                ),deriv = None
                            )
                        )
                    continue
            
                layer_result.append(
                    layer.activation(
                        np.dot(
                                layer_result[-1], layer.weight_layer
                            ),deriv = None
                        )
                )
                
            return layer_result