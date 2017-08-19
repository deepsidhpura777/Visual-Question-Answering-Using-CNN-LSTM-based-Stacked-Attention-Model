#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:08:43 2017

@author: deepsidhpura
"""
import re
from spacy.en import English
import cPickle as pickle

nlp = English()
def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n']


questions_train = open('/Users/deepsidhpura/Documents/VQA/data/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
questions_val = open('/Users/deepsidhpura/Documents/VQA/data/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
questions_test = open('/Users/deepsidhpura/Documents/VQA/data/questions_test2015.txt', 'r').read().decode('utf8').splitlines()

total_questions = questions_train + questions_val + questions_test
ctr = 0
word_to_index = {}

for i in total_questions:
    t = map(lambda x:x.strip('?\n'),i.split(' '))
    for j in t:
        if j not in word_to_index:
            word_to_index[str(j)] = ctr
            ctr += 1
            
f = open('original_word_index.pkl','wb+')
pickle.dump(word_to_index,f)
f.close()