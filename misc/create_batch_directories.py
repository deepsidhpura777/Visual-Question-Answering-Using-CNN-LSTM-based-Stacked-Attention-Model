#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:24:33 2017

@author: deepsidhpura
"""
def func(batches):
    for i in range(len(batches)):
        if len(batches[i]) < 128:
            batches[i].append(batches[i][0])

def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq


from PIL import Image
import os
import cPickle as pickle
from os import listdir
from os.path import isfile, join

mypath = '/Users/deepsidhpura/Downloads/val2014'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

original_images = pickle.load(open('original_val_list.pkl','rb+'))
original_images_batches = split_seq(original_images,732)
#func(original_images_batches)

### create a dictionary of batch number to list of unique images in that batch
#int(i[i.rfind('_')+1:i.find('.')])

dic_batch = {}

for i in range(len(original_images_batches)):
    dic_batch['batch_' + str(i+1)] = list(set(original_images_batches[i]))  #### order would change due to set, keep indices in mind !!!
   
dic_files = {}
for i in onlyfiles:
    dic_files[int(i[i.rfind('_')+1:i.find('.')])] = i   ##### this is an integer key dic
    
os.mkdir('/Users/deepsidhpura/Downloads/val_batches')
## Make a dir of name dic key
## read images from the path folder and save them in the newly created directory

ctr = 0
keys = list(dic_batch.keys())
for i in dic_batch:
    ctr += 1
    os.mkdir('/Users/deepsidhpura/Downloads/val_batches/' + i)
    for j in dic_batch[i]:
        temp_img = Image.open(mypath + '/' + dic_files[int(j)])
        temp_img.save('/Users/deepsidhpura/Downloads/val_batches/' + i + '/' + j + '.png')
    print 'Batch Number Done:',ctr


        








