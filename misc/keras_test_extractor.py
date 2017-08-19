#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:26:46 2017

@author: deepsidhpura
"""

def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
import cPickle as pickle

mypath = '/Users/deepsidhpura/Downloads/test2015'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

img_numbers_test = []
for i in onlyfiles:
    img_numbers_test.append(int(i[i.rfind('_')+1:i.find('.')]))
    
feat_dic_test = {}  ### Key = image numbers Value = 4096 dim vectors
                
base_model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
                
'''files_batches = split_seq(onlyfiles,38)
img_numbers_batches = split_seq(img_numbers_test,38)         
                
                
base_model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

feats_batches = []

flatten = lambda l: [item for sublist in l for item in sublist]
ctr = 0
for b in files_batches:
    temp = []
    ctr += 1
    for f in b:
        img_path = mypath + '/' + f
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = x[0]
        temp.append(x)
    temp = np.array(temp)
    print "Starting to Predict:",ctr
    fc2_features = model.predict(temp)
    feats_batches.append(fc2_features)
    print "Batch Done:",ctr
    
feats = flatten(feats_batches)
for i in range(len(img_numbers_test)):
    feat_dic_test[img_numbers_test[i]] = feats[i]'''
temp = []  
for i in range(len(onlyfiles)):
    img_path = mypath + '/' + onlyfiles[i]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x[0]
    temp.append(x)
temp = np.array(temp)    
fc2_features = model.predict(temp)
for i in range(len(onlyfiles)):
    feat_dic_test[img_numbers_test[i]] = fc2_features[i]

f = open('feats_test2015.pkl','wb+')
pickle.dump(feat_dic_test,f)