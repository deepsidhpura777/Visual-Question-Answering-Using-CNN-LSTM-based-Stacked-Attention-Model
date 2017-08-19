#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:14:19 2017

@author: deepsidhpura
"""

import numpy as np
import pandas as pd
### fixing time step size to 30 !

def get_word_vectors(data,batch_size,time_step,model): ### Data would be raw questions equivalent to the batch size
    matrix = np.zeros((batch_size,time_step,300))
    #model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    for i in range(len(data)):
        temp_words = map(lambda x:x.strip('?\n'),data[i].split(' '))
        temp_vectors = []
        for j in temp_words:
            if j in model:
                temp_vectors.append(model[j])
        temp_vectors = np.array(temp_vectors)
        for v in range(len(temp_vectors)):
            matrix[i,v] = temp_vectors[v]
    
    return matrix
                

def select_k_best_examples(questions_train,answers_train,images_train,max_answers):
    ### Find out the top frequent answers
    dic = {}
    indices = []
    new_questions_train = []
    new_answers_train = []
    new_images_train = []
    for i in answers_train:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
    
    sorted_keys = sorted(dic,key=dic.get,reverse=True)
    sorted_keys = sorted_keys[:max_answers]
    for i in range(len(answers_train)):
        if answers_train[i] in sorted_keys:
            indices.append(i)
            
    for i in indices:
        new_questions_train.append(questions_train[i])
        new_answers_train.append(answers_train[i])
        new_images_train.append(images_train[i])
    return new_questions_train,new_answers_train,new_images_train
    
    
def get_labels(answers): #### raw answers equivalent to the batch size  
    dum = pd.get_dummies(answers)
    dum = pd.DataFrame(dum.values)
    df = dum.idxmax(axis=1)
    matrix = np.array(df,dtype=np.int32)
    return matrix

def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq
    
def get_cnn_vectors(vectors_needed,cnn_model):
    cnn_features = cnn_model['feats'].T
    matrix = np.zeros((len(vectors_needed),cnn_features.shape[1]))
    for i in range(len(vectors_needed)):
        matrix[i] = cnn_features[vectors_needed[i]]
    
    tem = np.sqrt(np.sum(np.multiply(matrix,matrix), axis=1))  #### Performing normalization of each image feature independently !!!
    matrix = np.divide(matrix, np.transpose(np.tile(tem,(4096,1))))
    return matrix
        
    
              
        
        
