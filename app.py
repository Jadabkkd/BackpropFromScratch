from BackpropLab import model
from utils import activations,initializer,one_hot_encoding_csv

md = model()

md.add_hdl_layer(dense=10,activation=activations.Sigmoid,initializer=initializer.Xavier_initialization())
md.add_hdl_layer(
    dense=15,activation=activations.Tanh,initializer=initializer.Xavier_ReLu_initialization()
)

md.add_hdl_layer(
    dense=30,activation=activations.Tanh,initializer=initializer.Xavier_ReLu_initialization()
)

data = one_hot_encoding_csv('dataset/wine.csv',3,0,False,True)

md.add_weight(data['input_shape'],data['classes'])

md.train(data['data'],data['labels'])