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
import json

questions_test = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_test2015.txt', 'r').read().decode('utf8').splitlines()
images_test = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_test2015.txt', 'r').read().decode('utf8').splitlines()

f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()
f = open('/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/vgg_test_features.pkl','rb+')
cnn_model = pickle.load(f)

questions_test = extract_features.get_word_vectors(questions_test,len(questions_test),30,model)
questions_test = list(questions_test)

questions_test_batches = extract_features.split_seq(questions_test,2143)

vectors_needed = images_test[:]

images = extract_features.get_final_vectors(vectors_needed,cnn_model)
images = list(images)

images_test_batches = extract_features.split_seq(images,2143)


model = model_from_json(open("/home/rcf-proj2/jtr/dsidhpur/VQA/keras_vqa/lstm_cnn_glove_finalTest.json").read())
model.load_weights("/home/rcf-proj2/jtr/dsidhpur/VQA/keras_vqa/lstm_cnn_glove_finalTest_epoch_150.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

y_predict_all = []

for i in range(len(questions_test_batches)):
    X_q_test_batch = np.array(questions_test_batches[i])
    X_i_test_batch = np.array(images_test_batches[i])
    y_predict_batch = model.predict([X_i_test_batch,X_q_test_batch], verbose=1)
    y_predict_all.append(list(y_predict_batch))

f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/labels.pkl','rb+')
names_mapping = pickle.load(f)

flatten = lambda l: [item for sublist in l for item in sublist]
predictions = flatten(y_predict_all)

predictions = np.array(predictions)
predictions_classes = np.argmax(predictions,axis=1)
predictions_classes = list(predictions_classes)
predictions_names = map(lambda x:names_mapping[x],predictions_classes)
f = open('test_final_predictions.pkl','wb+')
pickle.dump(predictions_names,f)

questions_ids = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_id_test2015.txt', 'r').read().decode('utf8').splitlines()
list_json = []
for i in range(len(questions_ids)):
	d = {}
	d = {"question_id":int(questions_ids[i]),"answer":predictions_names[i]}
	list_json.append(d)

json.dump(list_json,open('test_predictions.json','wb+'))


