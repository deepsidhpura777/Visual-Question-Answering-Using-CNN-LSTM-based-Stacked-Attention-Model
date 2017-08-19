#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 15:01:30 2017

@author: deepsidhpura
"""
def func(batches):
    for i in range(len(batches)):
        if len(batches[i])<256:
            batches[i].append(batches[i][0])


import extract_features
import nn_architectures
import numpy as np
from random import shuffle
from keras.utils import generic_utils
import gensim
import time

##### Need to do something about the wordtovec model that gives the word vectors !!!
questions_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
answers_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_train2014.txt', 'r').read().decode('utf8').splitlines()
max_answers = 1000

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

questions,answers,images = extract_features.select_k_best_examples(questions_train,answers_train,images_train,max_answers)

answers = extract_features.get_labels(answers)
answers = list(answers)

questions = extract_features.get_word_vectors(questions,len(questions),30,model)
questions = list(questions)

#### A little wary of this syntax !
combined = zip(questions,answers,images)
shuffle(combined)

questions,answers,images = zip(*combined)
questions = list(questions)
answers = list(answers)
images = list(images)

#questions = np.array(questions)
##answers = np.array(answers)
#images = np.array(images)

print "Questions Length:",len(questions)

questions_batches = extract_features.split_seq(questions,842) ### Each batch will be appx 3365~~64 size, 1684~~128.
answers_batches = extract_features.split_seq(answers,842)
images_batches = extract_features.split_seq(images,842) #### To be used for combined cnn+rnn model.

func(questions_batches)
func(answers_batches)
func(images_batches)


num_epochs = 100  ### Changed from 100 for testing
num_hidden_units = 512
num_classes = max_answers
word_dim = 300
max_len = 30 #### Time step sequence length
dropout = 0.2

lstm_model = nn_architectures.lstm_network_ii(num_classes,num_hidden_units,max_len,word_dim,dropout)

json_string = lstm_model.to_json()
model_file_name = 'lstm_language_only'
open(model_file_name  + '.json', 'w').write(json_string)

model_save_interval = 5
time_step = 30

print "Model Started training, Bitch !!!"
#lstm_model.fit(questions,answers,batch_size=64,nb_epoch=num_epochs)

for i in range(num_epochs):
   
    now = time.time()
    #progbar = generic_utils.Progbar(len(questions))
    
    print "Epoch Started:",i
    ######## 600 batches chosen for testingb multi-gpu setting !!!
    #questions_batches = sorted(questions_batches,key=lambda x:len(x),reverse=True)[:630]
    #answers_batches = sorted(answers_batches,key=lambda x:len(x),reverse=True)[:630]	
    for j in range(len(questions_batches)):
        
        #temp = sorted(questions_batches[j],key=lambda x:len(x.split(' ')),reverse=True)
        #time_step = len(temp[0].split(' '))
        #X_batch = extract_features.get_word_vectors(questions_batches[j],len(questions_batches[j]),time_step,model)
        X_batch = np.array(questions_batches[j])
	y_batch = np.array(answers_batches[j])
        
	#print "Answer_Batches:",j,"Len:",len(answers_batches[j])
        loss = lstm_model.train_on_batch(X_batch,y_batch)
        
        #progbar.add(len(questions_batches[j]), values=[("train loss", loss)])
	print "Batch No:",j,"Y Size:","Training Loss",loss
    print
    print "Epoch Finished:",i
    later = time.time()
    print "Time per epoch in seconds:",int(later-now)
    if i % model_save_interval == 0:
        lstm_model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(i))
    
lstm_model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(i+1))
