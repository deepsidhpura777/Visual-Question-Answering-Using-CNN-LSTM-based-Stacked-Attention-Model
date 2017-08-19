#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:04:03 2017

@author: deepsidhpura
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 15:01:30 2017

@author: deepsidhpura
"""
def func(batches):
    for i in range(len(batches)):
        if len(batches[i]) < 128:
            batches[i].append(batches[i][0])


import extract_features
import nn_architectures
import numpy as np
from random import shuffle
import time
import scipy.io as sc
import cPickle as pickle

questions_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
answers_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_train2014.txt', 'r').read().decode('utf8').splitlines()
max_answers = 3000


f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()

#f = open('embeddings.pkl','rb+')
#model = pickle.load(f)
#f.close()

#f = open('original_word_index.pkl')
#word_to_index = pickle.load(f)
#f.close()

f = open('/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/resnet_train_features.pkl','rb+')
cnn_model = pickle.load(f)
f.close()


questions,answers,images = extract_features.select_k_best_examples(questions_train,answers_train,images_train,max_answers)

answers = extract_features.get_labels(answers)
answers = list(answers)

questions = extract_features.get_word_vectors(questions,len(questions),25,model)
questions = list(questions)

#questions = extract_features.get_questions_matrix(questions,len(questions),25,word_to_index)
#questions = list(questions)


vectors_needed = images[:]
    
images = extract_features.get_cnn_vectors(vectors_needed,cnn_model)
images = list(images)

questions_batches = extract_features.split_seq(questions,1798) ### Each batch will be appx  1798~~128.
answers_batches = extract_features.split_seq(answers,1798)
images_batches = extract_features.split_seq(images,1798)

func(questions_batches)
func(answers_batches)
func(images_batches)


num_epochs = 4  ### Changed from 120 for testing
num_hidden_units = 1024
num_classes = max_answers
word_dim = 300
img_dim = 2048
max_len = 25 #### Time step sequence length

lstm_resnet_model = nn_architectures.lstm_resnet_network(num_classes,num_hidden_units,max_len,word_dim,img_dim)
#lstm_resnet_model = nn_architectures.lstm_resnet_embedding_network(num_classes,num_hidden_units,max_len,word_dim,img_dim,model) 

json_string = lstm_resnet_model.to_json()
model_file_name = 'lstm_resnet_different'
open(model_file_name  + '.json', 'w').write(json_string)

model_save_interval = 20
time_step = 25

print "Model Started training, Bitch !!!"

for i in range(num_epochs):
   
    now = time.time()    
    for j in range(len(questions_batches)):
        
        X_q_batch = np.array(questions_batches[j])
	X_i_batch = np.array(images_batches[j])
        y_batch = np.array(answers_batches[j])
        loss = lstm_resnet_model.train_on_batch([X_i_batch,X_q_batch],y_batch)
        
    print "Batch No:",j,"Training Loss",loss
    print
    print "Epoch Finished:",i
    later = time.time()
    print "Time per epoch in seconds:",int(later-now)
    if i % model_save_interval == 0:
        lstm_resnet_model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(i))
    
lstm_resnet_model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(i+1))

