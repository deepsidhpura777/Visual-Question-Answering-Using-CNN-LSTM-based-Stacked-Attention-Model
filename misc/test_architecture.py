#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 19:18:25 2017

@author: deepsidhpura
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.recurrent import LSTM
from keras.layers import Merge
from keras.layers import Flatten
from keras.layers import Conv1D
import numpy as np
from sklearn.preprocessing import normalize


def get_maps(file_name): ##### takes npz file name as input and returns the maps
    temp_map = np.load(file_name)['arr_0'] ### (1000,14,14,2048)
    #temp_map = temp_map.reshape((1000,14,14,2048))  ### just for the sake of testing
    final_map = []
    for i in range(len(temp_map)):
        t = temp_map[i]
        l_map = [] 
        for j in range(t.shape[2]):
            x = t[:,:,j]
            x = x.flatten()
            l_map.append(x)
        l_map = np.array(l_map).T
        final_map.append(l_map)
    
    return final_map ##### List of (196,2048) maps of len big batch size (read from disk).
    
            
    
    




def weighted_average(outputs): ### takes a list of inputs from the attention model and the input model
    attention_outputs = outputs[0]
    image_outputs = outputs[1]
    
    weights_1 = attention_outputs[:,:,0][:,:-1]
    weights_2 = attention_outputs[:,:,1][:,:-1]
    
    l1 = []
    l2 = []
    
    for i in range(len(weights_1)): ### repeats batch size number of times
        t1 = weights_1[i]
        t2 = weights_2[i]
        t1 = normalize(t1,norm='l1')
        t2 = normalize(t2,norm='l1')
        temp_w_1 = np.reshape((len(t1),1)).dot(np.ones((1,image_outputs.shape[2])))
        temp_w_2 = np.reshape((len(t2),1)).dot(np.ones((1,image_outputs.shape[2])))
        temp_i = image_outputs[i]
        x1 = np.multiply(temp_w_1,temp_i)
        x2 = np.multiply(temp_w_2,temp_i)
        
        l1.append(x1)
        l2.append(x2)
    l1 = np.array(l1)  #### (None,392,1024)
    l2 = np.array(l2) #### (None,392,1024)
    
    l1 = l1.sum(axis=1) ##### (None,1024)
    l2 = l2.sum(axis=1) ##### (None,1024)
    
    l = np.concatenate((l1,l2),axis=1) ##### (None,2048)
    
    return l
    
    

def lstm_resnet_network(num_classes,num_hidden_units,max_len,word_dim,img_dim):
    image_model = Sequential()
    image_model.add(Reshape((196,img_dim),input_shape = (196,img_dim)))
    image_model.add(Reshape((196*(img_dim/num_hidden_units),num_hidden_units))) ##### (None,392,1024)

    lstm_model = Sequential()
    lstm_model.add(LSTM(num_hidden_units,activation='tanh',return_sequences=False,input_shape = (max_len,word_dim)))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Reshape((1,num_hidden_units)))
      
    attention_model = Sequential()
    attention_model.add(Merge([image_model,lstm_model],mode='concat',concat_axis=1))
    attention_model.add(Conv1D(512,1))
    attention_model.add(Activation('relu'))
    attention_model.add(Conv1D(2,1))
    attention_model.add(Activation('softmax'))
    
    average_model = Sequential()
    average_model.add(Merge([attention_model,image_model],mode=weighted_average,output_shape=(None,2048)))
    
    combined_model = Sequential()
    combined_model.add(Merge([average_model,lstm_model],mode='concat',concat_axis=1))
    combined_model.add(Flatten())
    combined_model.add(Dense(1024,init='glorot_uniform'))
    combined_model.add(Activation('relu'))

    combined_model.add(Dropout(0.5))
    combined_model.add(Dense(num_classes,init='glorot_uniform'))
    combined_model.add(Activation('softmax'))

    print 'Model Compilation started'
    combined_model.compile(loss='categorical_crossentropy',optimizer='adam') #### adam,rmsprop better than sgd for rnns
    print 'Model compilation done'
    
    
    #### Add sklearn l1 normalize to the output of the attention_model except the last one.
    #### There are 2 attention distributions of dimension 393, which act as weights for each of the 392 features
    #### Multiply it with 392 features each of dimension 1024
    #### Take the sum of these 392 features after multiplication across axis=0
    #### Concatenate 2 1024 dimensional vectors and proceed as normal !
    

    return combined_model     ##### call ([[[i_x,q_x],i_x],q_x],y)