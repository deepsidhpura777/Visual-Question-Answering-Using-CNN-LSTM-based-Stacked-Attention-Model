#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 09:58:42 2017

@author: deepsidhpura
"""

import numpy as np
import pandas as pd
import re
from spacy.en import English
nlp = English()


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
    
    return final_map




def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n']

### fixing time step size to 25 !

def get_questions_matrix(data,batch_size,time_step,word_to_index):  #### Returns matrix of batch size x time step
    vocab = len(word_to_index.keys())
    matrix = []
    for i in range(len(data)):
        temp_words = map(lambda x:x.strip('?\n'),data[i].split(' '))
        temp_vectors = []
        temp_vectors = [vocab] * time_step
        for j in range(len(temp_words)):
            if j < time_step:
                temp_vectors[j] = word_to_index[temp_words[j]]
        matrix.append(temp_vectors[:])
    return np.array(matrix)



def get_word_vectors(data,batch_size,time_step,model): ### Data would be raw questions equivalent to the batch size
    matrix = np.zeros((batch_size,time_step,300))
    bad_chars = '?\',.!"\n'
    for i in range(len(data)):
        #temp_words = map(lambda x:x.strip('?\n'),data[i].split(' '))
	#temp_words = nlp(data[i])
        temp_words = str(data[i]).translate(None,bad_chars).split(' ')
        if '' in temp_words:
 	    temp_words.remove('')
        temp_vectors = []
        for j in temp_words:
            if j in model:
                temp_vectors.append(model[j])
        temp_vectors = np.array(temp_vectors)
	for v in range(len(temp_vectors)):
	    if v < time_step:	
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
    matrix = np.array(dum,dtype=np.float64)
    return matrix

def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq
              
              
        
def get_cnn_vectors(vectors_needed,cnn_model): 
    matrix = np.zeros((len(vectors_needed),2048))
    for i in range(len(vectors_needed)):
    	matrix[i] = cnn_model[vectors_needed[i]]
    tem = np.sqrt(np.sum(np.multiply(matrix,matrix), axis=1))  #### Performing normalization of each image feature independently !!!
    matrix = np.divide(matrix, np.transpose(np.tile(tem,(2048,1))))
    return matrix

def get_final_vectors(vectors_needed,cnn_model): 
    matrix = np.zeros((len(vectors_needed),4096))
    for i in range(len(vectors_needed)):
    	matrix[i] = cnn_model[vectors_needed[i]]
    return matrix  
    

def extract_labels(batch,labels):
    x = np.zeros((len(batch),len(labels)))
    for i in range(len(batch)):
        index = labels.index(batch[i])
        x[i,index] = 1.0

    return x  

def normalize(matrix):
    for i in range(len(matrix)):
	m = matrix[i]
        tem = np.sqrt(np.sum(np.multiply(m,m), axis=1))  #### Performing normalization of each image feature independently !!!
        m = np.divide(m, np.transpose(np.tile(tem,(2048,1))))
        matrix[i] = m

    return matrix
