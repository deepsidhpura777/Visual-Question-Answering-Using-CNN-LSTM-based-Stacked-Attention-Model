#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 12:52:00 2017

@author: deepsidhpura
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape,Merge
from keras.layers.recurrent import LSTM
from multi_gpu import make_parallel

def lstm_network(num_classes,num_hidden_units,max_len,word_dim,dropout):
    num_hidden_units = num_hidden_units
    dropout = dropout
    max_len = max_len #### Yet to test if it will work for variable lengths, no it doesnt work. Will have to fix it to 30
    word_dim = word_dim
    num_classes = num_classes
    model = Sequential()
    model.add(LSTM(num_hidden_units,activation='tanh',return_sequences=True,input_shape = (max_len,word_dim)))
    model.add(Dropout(dropout))
    model.add(LSTM(num_hidden_units,return_sequences=False,activation='tanh')) ### return_seq indicates if we require output at each time step.
    model.add(Dense(num_classes,init='uniform'))
    model.add(Activation('softmax'))
    
    model = make_parallel(model, 2)
    print 'Model Compilation started'
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop') #### rmsprop better than sgd for rnns
    print 'Model compilation done'
    
    return model
    
def lstm_network_ii(num_classes,num_hidden_units,max_len,word_dim,dropout):
    num_hidden_units = num_hidden_units
    dropout = dropout
    max_len = max_len #### Yet to test if it will work for variable lengths, no it doesnt work. Will have to fix it to 30
    word_dim = word_dim
    num_classes = num_classes
    model = Sequential()
    model.add(LSTM(num_hidden_units,activation='tanh',return_sequences=True,input_shape = (max_len,word_dim)))
    model.add(Dropout(dropout))
    model.add(LSTM(num_hidden_units,return_sequences=True,activation='tanh'))
    model.add(Dropout(dropout))
    model.add(LSTM(num_hidden_units,return_sequences=True,activation='tanh'))
    model.add(Dropout(dropout))
    model.add(LSTM(num_hidden_units,return_sequences=False,activation='tanh')) ### return_seq indicates if we require output at each time step.
    model.add(Dropout(dropout))
    model.add(Dense(num_classes,init='uniform'))
    model.add(Activation('softmax'))

    model = make_parallel(model, 2)
    print 'Model Compilation started'
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop') #### rmsprop better than sgd for rnns
    print 'Model compilation done'

    return model

def lstm_cnn_network(num_classes,num_hidden_units,max_len,word_dim,dropout,img_dim,num_fully_units):
    image_model = Sequential()
    image_model.add(Reshape((img_dim,),input_shape = (img_dim,)))
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(num_hidden_units,activation='tanh',return_sequences=True,input_shape = (max_len,word_dim)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(num_hidden_units,return_sequences=True,activation='tanh')) ### return_seq indicates if we require output at each time step.
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(num_hidden_units,return_sequences=False,activation='tanh'))
    lstm_model.add(Dropout(0.2))
    #lstm_model.add(LSTM(num_hidden_units,return_sequences=False,activation='tanh'))
    #lstm_model.add(Dropout(dropout))
    
    combined_model = Sequential()
    combined_model.add(Merge([image_model,lstm_model],mode='concat',concat_axis=1))
    combined_model.add(Dense(num_fully_units,init='glorot_uniform'))
    combined_model.add(Activation('tanh'))
    combined_model.add(Dropout(dropout))
    combined_model.add(Dense(num_fully_units,init='glorot_uniform'))
    combined_model.add(Activation('tanh'))
    combined_model.add(Dropout(dropout))
    #combined_model.add(Dense(num_fully_units,init='glorot_uniform'))
    #combined_model.add(Activation('tanh'))
    #combined_model.add(Dropout(dropout))
    combined_model.add(Dense(num_classes,init='glorot_uniform'))
    combined_model.add(Activation('softmax'))
    
    combined_model = make_parallel(combined_model, 2)         
    print 'Model Compilation started'
    combined_model.compile(loss='categorical_crossentropy',optimizer='rmsprop') #### rmsprop better than sgd for rnns
    print 'Model compilation done'
    
    return combined_model 
