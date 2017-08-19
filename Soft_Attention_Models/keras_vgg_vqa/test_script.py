# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import scipy.io as sc

cnn_features = sc.loadmat('vgg_feats.mat')
cnn_features = cnn_features['feats']

f = open('/Users/student/Downloads/data/preprocessed/questions_train2014.txt')
questions_train = f.readlines()

f = open('/Users/student/Downloads/data/preprocessed/questions_id_train2014.txt')
questions_train_id = f.readlines()

f = open('/Users/student/Downloads/data/preprocessed/images_train2014.txt')
images_train = f.readlines()

