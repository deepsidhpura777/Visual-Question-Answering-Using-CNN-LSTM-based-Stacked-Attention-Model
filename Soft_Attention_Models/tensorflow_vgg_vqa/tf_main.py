#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:36:40 2017

@author: deepsidhpura
"""

def func(batches):
    for i in range(len(batches)):
        if len(batches[i])<501:
            batches[i].append(batches[i][0])


import tf_extract_features
from tf_architectures import NN
import numpy as np
import time
import scipy.io as sc
import cPickle as pickle
import tensorflow as tf
import os

##### Need to do something about the wordtovec model that gives the word vectors !!!
questions_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
answers_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_train2014.txt', 'r').read().decode('utf8').splitlines()
max_answers = 1000

f = open('glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()

#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
cnn_model = sc.loadmat('/home/rcf-proj2/jtr/dsidhpur/VQA/data/vgg_feats.mat')


questions,answers,images = tf_extract_features.select_k_best_examples(questions_train,answers_train,images_train,max_answers)

answers = tf_extract_features.get_labels(answers)
answers = list(answers)

questions = tf_extract_features.get_word_vectors(questions,len(questions),30,model)
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
    
images = tf_extract_features.get_cnn_vectors(vectors_needed,cnn_model)
images = list(images)

questions_batches = tf_extract_features.split_seq(questions,430) ### Each batch will be appx 3365~~64 size, 1684~~128.
answers_batches = tf_extract_features.split_seq(answers,430)   ###### Changed the answers to column numbers in tf_extract_features !!!!
images_batches = tf_extract_features.split_seq(images,430) #### To be used for combined cnn+rnn model.

func(questions_batches)
func(answers_batches)
func(images_batches)

learning_rate = 0.0003		
learning_rate_decay_start = -1		# at what iteration to start decaying learning rate? (-1 = dont)
batch_size = 501  ### Changed from 256		
word_dim = 300	
rnn_units = 512				
rnn_layers = 2				
image_dim = 4096
hidden_units = 1024 
num_classes = 1000			
img_norm = 1				
decay_factor = 0.99997592083
dropout = 0.5

# path to save the mode
#checkpoint_path = 'model_save/'

gpu_id = 0
num_epochs = 300 #### Changed from 150 !!!
time_step = 30
model_save_interval = 20

def train():
    model = NN(rnn_units,rnn_layers,batch_size,word_dim,image_dim,hidden_units,time_step,dropout,num_classes)
    
    tf_loss, tf_image, tf_question, tf_answer = model.build_network_train()
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=10)
    
    training_vars = tf.trainable_variables()
    lr = tf.Variable(learning_rate)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=lr) 
    
    gradients = optimizer.compute_gradients(tf_loss,training_vars)
    clipped_gradients = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gradients]
    train_optimizer = optimizer.apply_gradients(clipped_gradients)
    
    tf.initialize_all_variables().run()
    
    print "Model Started training, Bitch !!!"
    
    for i in range(num_epochs):
       
        now = time.time()    
        for j in range(len(questions_batches)):
            
            X_q_batch = np.array(questions_batches[j])
            X_i_batch = np.array(images_batches[j]) ## Need to normalize image features !
            y_batch = np.array(answers_batches[j])
            
            _,loss = sess.run([train_optimizer,tf_loss],feed_dict={tf_image : X_i_batch, tf_question : X_q_batch,tf_answer : y_batch})
            #print "Loss:",loss
            current_learning_rate = lr * decay_factor
            lr.assign(current_learning_rate).eval()
        print "Training Loss:",loss
        print
        print "Epoch Finished:",i
        later = time.time()
        print "Time per epoch in seconds:",int(later-now)
        if i % model_save_interval == 0:
            saver.save(sess,'model', global_step=i)
            
    print "Finally saving the model !!"
    saver.save(sess,'model', global_step=i)
    
with tf.device('/gpu:'+str(0)):   ####### Right now just trainingb on one gpu to see if everything is working fine !!!!
    train()
