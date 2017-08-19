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
        if len(batches[i]) < 512:
            batches[i].append(batches[i][0])


import extract_features
import nn_architectures
import numpy as np
from random import shuffle
import time
import scipy.io as sc
import cPickle as pickle
##### Need to do something about the wordtovec model that gives the word vectors !!!
questions_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
answers_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_train2014.txt', 'r').read().decode('utf8').splitlines()
max_answers = 1000

questions_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
answers_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_val2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_val2014_all.txt', 'r').read().decode('utf8').splitlines()

questions_train = questions_train + questions_val
answers_train = answers_train + answers_val
images_train = images_train + images_val

f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()

#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
cnn_model = sc.loadmat('/home/rcf-proj2/jtr/dsidhpur/VQA/data/vgg_feats.mat')


questions,answers,images = extract_features.select_k_best_examples(questions_train,answers_train,images_train,max_answers)

answers = extract_features.get_labels(answers)
answers = list(answers)

questions = extract_features.get_word_vectors(questions,len(questions),30,model)
questions = list(questions)

f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/coco_vgg_IDMap.txt','rb+')
id_maps = f.readlines()
id_maps = map(lambda x:x.split(' '),id_maps)
dic_m = {}
for i in id_maps:
    dic_m[i[0]] = int(i[1])
    
vectors_needed = []
for i in images:
    vectors_needed.append(dic_m[i.strip('\n')])
    
images = extract_features.get_cnn_vectors(vectors_needed,cnn_model)
images = list(images)

#### A little wary of this syntax !
combined = zip(questions,answers,images)
shuffle(combined)

questions,answers,images = zip(*combined)
questions = list(questions)
answers = list(answers)
images = list(images) ### cnn feature vector now!


questions_batches = extract_features.split_seq(questions,626) ### Each batch will be appx 3365~~64 size, 1684~~128.
answers_batches = extract_features.split_seq(answers,626)
images_batches = extract_features.split_seq(images,626) #### To be used for combined cnn+rnn model.

func(questions_batches)
func(answers_batches)
func(images_batches)


num_epochs = 150  ### Changed from 100 for testing
num_hidden_units = 512
num_fully_units = 1024
num_classes = max_answers
word_dim = 300
img_dim = 4096
max_len = 30 #### Time step sequence length
dropout = 0.3

lstm_cnn_model = nn_architectures.lstm_cnn_network(num_classes,num_hidden_units,max_len,word_dim,dropout,img_dim,num_fully_units)

json_string = lstm_cnn_model.to_json()
model_file_name = 'lstm_cnn_glove_finalTest'
open(model_file_name  + '.json', 'w').write(json_string)

model_save_interval = 10
time_step = 30

print "Model Started training, Bitch !!!"

for i in range(num_epochs):
   
    now = time.time()    
    for j in range(len(questions_batches)):
        
        X_q_batch = np.array(questions_batches[j])
	X_i_batch = np.array(images_batches[j])
        y_batch = np.array(answers_batches[j])
        loss = lstm_cnn_model.train_on_batch([X_i_batch,X_q_batch],y_batch)
        
    print "Batch No:",j,"Y Size:","Training Loss",loss
    print
    print "Epoch Finished:",i
    later = time.time()
    print "Time per epoch in seconds:",int(later-now)
    if i % model_save_interval == 0:
        lstm_cnn_model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(i))
    
lstm_cnn_model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(i+1))

