#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 22:48:22 2017

@author: deepsidhpura
"""

import tf_extract_features
from tf_architectures import NN
import numpy as np
import scipy.io as sc
import cPickle as pickle
import tensorflow as tf


questions_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
answers_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_val2014_all.txt', 'r').read().decode('utf8').splitlines()
images_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_val2014_all.txt', 'r').read().decode('utf8').splitlines()

#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
f = open('glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()
cnn_model = sc.loadmat('/home/rcf-proj2/jtr/dsidhpur/VQA/data/vgg_feats.mat')

answers_val = map(lambda x:x.split(';'),answers_val)

questions_val = tf_extract_features.get_word_vectors(questions_val,len(questions_val),30,model)
questions_val = list(questions_val)

questions_val_batches = tf_extract_features.split_seq(questions_val,83) ##### Batch_Size = 1464

f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/coco_vgg_IDMap.txt','rb+')
id_maps = f.readlines()
id_maps = map(lambda x:x.split(' '),id_maps)
dic_m = {}
for i in id_maps:
    dic_m[i[0]] = int(i[1])

vectors_needed = []
for i in images_val:
    vectors_needed.append(dic_m[i.strip('\n')])

images = tf_extract_features.get_cnn_vectors(vectors_needed,cnn_model)
images = list(images)

images_val_batches = tf_extract_features.split_seq(images,83)


batch_size = 1464	
word_dim = 300	
rnn_units = 512				
rnn_layers = 2				
image_dim = 4096
hidden_units = 1024 
num_classes = 1000			
img_norm = 1				
decay_factor = 0.99997592083
dropout = 0  ###### Dropout should be zero for testing as we need all the units !!!

time_step = 30
gpu_id = 0

def test():
    
    model = NN(rnn_units,rnn_layers,batch_size,word_dim,image_dim,hidden_units,time_step,dropout,num_classes)
    
    tf_final_answer,tf_image,tf_question = model.build_network_test()
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess,'tf_weights/model-80')
    y_predict_all = []

    for i in range(len(questions_val_batches)):
        X_q_val_batch = np.array(questions_val_batches[i])
        X_i_val_batch = np.array(images_val_batches[i])
        
        predicted_ans = sess.run(tf_final_answer,feed_dict={tf_image:X_i_val_batch,tf_question:X_q_val_batch})
        class_ans = np.argmax(predicted_ans,axis=1)
        y_predict_all.append(list(class_ans))
        
    f = open('tf_predictions.pkl','wb+')
    pickle.dump(y_predict_all,f)
with tf.device('/gpu:'+str(0)):   ####### Right now just training on one gpu to see if everything is working fine !!!!
    test()
        
        
    
