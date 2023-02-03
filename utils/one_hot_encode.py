import numpy as np
import csv

def one_hot_encoding_csv(csvf:str, classes:int, index_label:int, index_from_zero:bool = True, shuffle:bool = False):
    data_list = []
    
    with open(csvf, 'r',newline='') as reader:
        for i in csv.reader(reader):
            data_list.append([float(x) for x in i])
    
    #convert list to array
    data_array = np.array(data_list)
    
    if shuffle: np.random.shuffle(data_list)    
    #zeros array for one hot encoding
    one_hot_arr = np.zeros( [len(data_array),classes] )
    
    #set up one hot for one hot zeros array
    for data_idx, one_h_idx in enumerate(data_array[:,index_label]):
        
        if index_from_zero: 
            one_hot_arr[ data_idx, int(one_h_idx) ] = 1 
            continue
        
        one_hot_arr[ data_idx, int(one_h_idx) - 1 ] = 1
        
    data_array = np.delete(data_array,index_label,1)

    return {
        'classes':classes,
        'data': data_array,
        'one_hot_arr': one_hot_arr
    }
        