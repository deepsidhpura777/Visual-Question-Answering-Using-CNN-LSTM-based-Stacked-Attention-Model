#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:10:47 2017

@author: deepsidhpura
"""

def accuracy(predictions,answers):
    ctr = 0
    points = 0
    for i in range(len(predictions)):
        p = predictions[i]
        a_list = answers[i]
        for a in a_list:
            if p==a:
                ctr += 1
        if ctr >=3:
            points += 1
	else:
	    points += float(ctr)/3
        ctr = 0
    return float(points)/len(answers)
                


import cPickle as pickle


f = open('labels.pkl','rb+')
names_mapping = pickle.load(f)



f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/tf_predictions.pkl','rb+')

predictions = pickle.load(f)
flatten = lambda l: [item for sublist in l for item in sublist]
predictions = flatten(predictions)
 
predictions_names = map(lambda x:names_mapping[x],predictions)

answers_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_val2014_all.txt', 'r').read().decode('utf8').splitlines()
answers_val = map(lambda x:x.split(';'),answers_val)

print "Model Accuracy:",accuracy(predictions_names,answers_val)
