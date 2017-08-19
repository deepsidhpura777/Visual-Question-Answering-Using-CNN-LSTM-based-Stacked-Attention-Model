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
import shutil
from shutil import copyfile

mypath = '/Users/deepsidhpura/Downloads/train2014'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#original_images = pickle.load(open('original_val_list.pkl','rb+'))
#original_images_batches = split_seq(original_images,732)
#func(original_images_batches)

original_images_batches = pickle.load(open('train/images_batches_list.pkl','rb+'))

### create a dictionary of batch number to list of unique images in that batch
#int(i[i.rfind('_')+1:i.find('.')])

dic_batch = {}

for i in range(len(original_images_batches)):
    dic_batch['batch_' + str(i+1)] = list(set(original_images_batches[i]))  #### order would change due to set, keep indices in mind !!!
   
dic_files = {}
for i in onlyfiles:
    dic_files[int(i[i.rfind('_')+1:i.find('.')])] = i   ##### this is an integer key dic
    
os.mkdir('/Users/deepsidhpura/Downloads/train_batches')
## Make a dir of name dic key
## read images from the path folder and save them in the newly created directory

ctr = 0
keys = list(dic_batch.keys())
for i in dic_batch:
    ctr += 1
    os.mkdir('/Users/deepsidhpura/Downloads/train_batches/' + str(i))
    for j in dic_batch[i]:
        #temp_img = Image.open(mypath + '/' + dic_files[int(j)])
        #temp_img.save('/Users/deepsidhpura/Downloads/train_batches/' + str(i) + '/' + str(j) + '.png')
        source = mypath + '/' + dic_files[int(j)]
        destination = '/Users/deepsidhpura/Downloads/train_batches/' + str(i) + '/' + str(j) + '.png'
        copyfile(source,destination)
    print 'Batch Number Done:',ctr

'''
mypath = '/home/rcf-proj2/jtr/dsidhpur/final_features/train_batches_zip'
from os import listdir
from os.path import isfile, join

mega_onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mega_onlyfiles = sorted(mega_onlyfiles,key=lambda x:int(x[x.find('_')+1:x.find('.')]))

mega_only_files[:50]

from PIL import Image
from resizeimage import resizeimage
i = Image.open(open('new.png','rb+'))
cover = resizeimage.resize_thumbnail(i, [400,400])
c = resizeimage.resize_crop(cover,[299,299])
im = cv2.cvtColor(np.array(c,dtype=np.float32), cv2.COLOR_RGB2BGR)


'''      








