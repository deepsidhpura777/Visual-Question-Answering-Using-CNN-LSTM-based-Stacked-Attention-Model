#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:19:31 2017

@author: deepsidhpura
"""
#### Need to change code according to the tensorflow version !!!

import tensorflow as tf
import numpy as np
#from tensorflow.models.rnn import rnn_cell ### in contrib module for the newer version

class NN():
    def __init__(self,rnn_units,rnn_layers,batch_size,word_dim,image_dim,hidden_units,time_step,dropout,num_classes):
        
        self.rnn_units = rnn_units ### number of RNN hidden units
        self.rnn_layers = rnn_layers
        self.batch_size = batch_size
        self.word_dim = word_dim
        self.image_dim = image_dim
        self.hidden_units = hidden_units
        self.time_step = time_step
        self.dropout = dropout
        self.num_classes = num_classes
        
        ### I wont need the question embedding
        
        ########## LSTM Layer Specifications !!!! #######################
        self.lstm_1 = tf.nn.rnn_cell.LSTMCell(rnn_units,word_dim,use_peepholes='True')
        self.lstm_1.dropout = tf.nn.rnn_cell.DropoutWrapper(self.lstm_1,output_keep_prob = 1-self.dropout)
                         
        self.lstm_2 = tf.nn.rnn_cell.LSTMCell(rnn_units,rnn_units,use_peepholes='True')
        self.lstm_2.dropout = tf.nn.rnn_cell.DropoutWrapper(self.lstm_2,output_keep_prob = 1-self.dropout)
                                                                    
        self.stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([self.lstm_1.dropout,self.lstm_2.dropout])
        
        ########### Fully Connected layer to the LSTM output weight specifications #####################
        self.embed_rnn_W = tf.Variable(tf.random_uniform([2 * rnn_units * rnn_layers,self.hidden_units],-0.08,0.08),name='embed_rnn_W')
        self.embed_rnn_b = tf.Variable(tf.random_uniform([self.hidden_units],-0.08,0.08),name='embed_rnn_b')
        
        ########## Fully Connected layer to the Image output weight specifications ###################
        self.embed_image_W = tf.Variable(tf.random_uniform([self.image_dim,self.hidden_units],-0.08,0.08),name='embed_image_W')
        self.embed_image_b = tf.Variable(tf.random_uniform([self.hidden_units],-0.08,0.08),name='embed_image_b')
        
        ########## Final Softmax weights after the point wise multiplication layer sepcifications ##########
        self.embed_softmax_W = tf.Variable(tf.random_uniform([self.hidden_units,self.num_classes],-0.08,0.08),name='embed_softmax_W')
        self.embed_softmax_b = tf.Variable(tf.random_uniform([self.num_classes],-0.08,0.08),name='embed_softmax_b')
        
    
    def build_network_train(self):
        image = tf.placeholder(tf.float32,[self.batch_size,self.image_dim]) ##### Placeholder for the images
        question = tf.placeholder(tf.float32,[self.batch_size,self.time_step,self.word_dim]) ###### Place holder for questions. It incldues the word embeddings !!!
        answer = tf.placeholder(tf.int32,[self.batch_size,])  ####### Some doubts regarding it, whether it has to be batch size X classes or number denoting which column of the logit is the class
        
        #### Here I am directly calling rnn object with all the inputs instead of the stacked lstm object per time step
        init_state = tf.zeros([self.batch_size,self.stacked_lstm.state_size])
        loss = 0.0
        output,state = tf.nn.dynamic_rnn(self.stacked_lstm,question,initial_state=init_state)  ##### Some doubts if it would work or not !!!
        
        rnn_dropout = tf.nn.dropout(state,1-self.dropout)
        rnn_linear = tf.nn.xw_plus_b(rnn_dropout,self.embed_rnn_W,self.embed_rnn_b)
        rnn_emb = tf.tanh(rnn_linear)
        
        image_dropout = tf.nn.dropout(image,1-self.dropout)
        image_linear = tf.nn.xw_plus_b(image_dropout, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(image_linear)
        
        pointwise_mul = tf.mul(rnn_emb,image_emb)
        pointwise_mul_dropout = tf.nn.dropout(pointwise_mul,1-self.dropout)
        pointwise_mul_embed = tf.nn.xw_plus_b(pointwise_mul_dropout,self.embed_softmax_W,self.embed_softmax_b)
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(pointwise_mul_embed,answer) #### wary of the answer dimension !!!!!
        
        loss = tf.reduce_mean(cross_entropy)

        return loss,image,question,answer                                                            
        
        
    def build_network_test(self):
        image = tf.placeholder(tf.float32,[self.batch_size,self.image_dim]) #### batch size is different here than the training one !!!
        question = tf.placeholder(tf.float32,[self.batch_size,self.time_step,self.word_dim])
        
        init_state = tf.zeros([self.batch_size,self.stacked_lstm.state_size])
        
        output,state = tf.nn.dynamic_rnn(self.stacked_lstm,question,initial_state=init_state)
        
        rnn_dropout = tf.nn.dropout(state,1-self.dropout)
        rnn_linear = tf.nn.xw_plus_b(rnn_dropout,self.embed_rnn_W,self.embed_rnn_b)
        rnn_emb = tf.tanh(rnn_linear)
        
        image_dropout = tf.nn.dropout(image,1-self.dropout)
        image_linear = tf.nn.xw_plus_b(image_dropout, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(image_linear)
        
        pointwise_mul = tf.mul(rnn_emb,image_emb)
        pointwise_mul_dropout = tf.nn.dropout(pointwise_mul,1-self.dropout)
        pointwise_mul_embed = tf.nn.xw_plus_b(pointwise_mul_dropout,self.embed_softmax_W,self.embed_softmax_b)
        
        final_answer = tf.nn.xw_plus_b(pointwise_mul_dropout,self.embed_softmax_W,self.embed_softmax_b) ### Same as the pointwise_mul_embed. Just for distinction !!!
        
        return final_answer,image,question    
        
        
