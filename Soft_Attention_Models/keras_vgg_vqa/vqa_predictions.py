#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:39:52 2017

@author: deepsidhpura
"""
def func(batches,indices):
    for i in range(len(batches)):
        if len(batches[i])<256:
            batches[i].append(batches[i][0])
	    indices.append(i)
import extract_features
import nn_architectures
import numpy as np
from keras.utils import generic_utils
import gensim
from keras.models import model_from_json
import cPickle as pickle
import tensorflow as tf
from multi_gpu import make_parallel
import scipy.io as sc

questions_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
answers_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_val2014_all.txt', 'r').read().decode('utf8').splitlines()
images_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_val2014_all.txt', 'r').read().decode('utf8').splitlines()

#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
f = open('glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()
cnn_model = sc.loadmat('/home/rcf-proj2/jtr/dsidhpur/VQA/data/vgg_feats.mat')

answers_val = map(lambda x:x.split(';'),answers_val)

questions_val = extract_features.get_word_vectors(questions_val,len(questions_val),30,model)
questions_val = list(questions_val)

questions_val_batches = extract_features.split_seq(questions_val,83)

f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/coco_vgg_IDMap.txt','rb+')
id_maps = f.readlines()
id_maps = map(lambda x:x.split(' '),id_maps)
dic_m = {}
for i in id_maps:
    dic_m[i[0]] = int(i[1])

vectors_needed = []
for i in images_val:
    vectors_needed.append(dic_m[i.strip('\n')])

images = extract_features.get_cnn_vectors(vectors_needed,cnn_model)
images = list(images)

images_val_batches = extract_features.split_seq(images,83)

indices = [] #### denotes where I have added a repeated value so that I would remove it while calculating accuracy, popping the last value!!
#func(questions_val_batches,indices)


model = model_from_json(open("/home/rcf-proj2/jtr/dsidhpur/VQA/lstm_cnn_combined_glove.json").read())
model.load_weights("/home/rcf-proj2/jtr/dsidhpur/VQA/lstm_cnn_combined_glove_epoch_100.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

y_predict_all = []

for i in range(len(questions_val_batches)):
    X_q_val_batch = np.array(questions_val_batches[i])
    X_i_val_batch = np.array(images_val_batches[i])
    y_predict_batch = model.predict([X_i_val_batch,X_q_val_batch], verbose=1)
    y_predict_all.append(list(y_predict_batch))

f = open('glove_predictions.pkl','wb+')
pickle.dump(y_predict_all,f)
