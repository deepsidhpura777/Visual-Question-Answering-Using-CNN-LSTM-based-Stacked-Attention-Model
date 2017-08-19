#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 20:02:50 2017

@author: deepsidhpura
"""
###### Script matches image numbers to the corresponding questions asked !!!
###### { image_numbers : [list of questions] }

import cPickle as pickle

questions_train = open('/Users/deepsidhpura/Documents/VQA/data/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
questions_val = open('/Users/deepsidhpura/Documents/VQA/data/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
questions_test = open('/Users/deepsidhpura/Documents/VQA/data/questions_test2015.txt', 'r').read().decode('utf8').splitlines()

images_train = open('/Users/deepsidhpura/Documents/VQA/data/images_train2014.txt', 'r').read().decode('utf8').splitlines()
images_val = open('/Users/deepsidhpura/Documents/VQA/data/images_val2014_all.txt', 'r').read().decode('utf8').splitlines()
images_test = open('/Users/deepsidhpura/Documents/VQA/data/images_test2015.txt', 'r').read().decode('utf8').splitlines()

train_map = {}
val_map = {}
test_map = {}

for i in range(len(images_train)):
    if images_train[i] not in train_map:
        train_map[images_train[i]] = [questions_train[i]]
    else:
        train_map[images_train[i]].append(questions_train[i])
        

for i in range(len(images_val)):
    if images_val[i] not in val_map:
        val_map[images_val[i]] = [questions_val[i]]
    else:
        val_map[images_val[i]].append(questions_val[i])    
        
for i in range(len(images_test)):
    if images_test[i] not in test_map:
        test_map[images_test[i]] = [questions_test[i]]
    else:
        test_map[images_test[i]].append(questions_test[i])
        

'''f = open('train_map.pkl','wb+')
pickle.dump(train_map,f)
pickle.close()   '''     

