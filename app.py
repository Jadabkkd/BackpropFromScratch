from BackpropLab import model
from utils import activations,initializer,one_hot_encoding_csv

md = model()

md.add_layer(dense=10,activation=activations.Sigmoid,initializer=initializer.Xavier_initialization)

data = one_hot_encoding_csv('dataset/wine.csv',3,0,False,True)
print(data)