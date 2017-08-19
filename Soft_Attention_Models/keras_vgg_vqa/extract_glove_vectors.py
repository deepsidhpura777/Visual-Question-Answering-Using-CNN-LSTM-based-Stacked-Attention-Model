#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:31:51 2017

@author: deepsidhpura
"""
import numpy as np
import cPickle as pickle

def loadGloveModel(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = np.array(embedding)
    print "Done.",len(model)," words loaded!"
    f.close()
    return model


model = loadGloveModel("glove.840B.300d.txt")
f = open('glove_vectors.pkl','wb+')
pickle.dump(model,f)
f.close()

