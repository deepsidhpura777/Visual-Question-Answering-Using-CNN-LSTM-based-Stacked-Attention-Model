#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:23:59 2017

@author: deepsidhpura
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:26:46 2017

@author: deepsidhpura
"""
import cv2
import numpy as np
import copy

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation
from keras.layers import merge
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializations
from keras.engine import Layer, InputSpec
from keras import backend as K

import sys
sys.setrecursionlimit(3000)

class Scale(Layer):
    '''Custom Layer for ResNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, 1, 1, name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, kernel_size, kernel_size, name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = merge([x, input_tensor],mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, 1, 1,subsample=strides, name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1', bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = merge([x, shortcut],mode='sum',name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet152_model(weights_path=None):
    '''Instantiate the ResNet152 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(448, 448, 3), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(3, 224, 224), name='data')
            
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    #x_fc = Flatten()(x_fc)
    #x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)
    
    # load weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model

import cPickle as pickle
import sys
from os import listdir
from os.path import isfile, join
import zipfile
import shutil
import os


def get_maps(temp_map): ##### takes npz file name as input and returns the maps
     ### (1000,14,14,2048)
    #temp_map = temp_map.reshape((1000,14,14,2048))  ### just for the sake of testing
    final_map = []
    for i in range(len(temp_map)):
        t = temp_map[i]
        l_map = []
        for j in range(t.shape[2]):
            x = t[:,:,j]
            x = x.flatten()
            l_map.append(x)
        l_map = np.array(l_map).T
        final_map.append(l_map)

    return np.array(final_map)

def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq

def func(batches):
    for i in range(len(batches)):
        if len(batches[i]) < 128:
            batches[i].append(batches[i][0])

original_images = pickle.load(open('original_images_list.pkl','rb+'))
original_images_batches = split_seq(original_images,1798)
func(original_images_batches)

#original_images = pickle.load(open('original_val_list.pkl','rb+'))
#original_images_batches = split_seq(original_images,732)

weights_path = 'resnet152_weights_tf.h5'
model = resnet152_model(weights_path)

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


mypath = '/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/train_batches_zip'

mega_onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mega_onlyfiles = sorted(mega_onlyfiles,key=lambda x:int(x[x.find('_')+1:x.find('.')]))

mega_onlyfiles_split = split_seq(mega_onlyfiles,35)  ####### Might need to change to 70

for split in mega_onlyfiles_split:
    batch_list = []
    batch_images = []
    batch_mappings = {}

    for zip_file in split:
        ######### Zipping procedure ###################
        zip_ref = zipfile.ZipFile(mypath + '/' + zip_file, 'r')
        zip_ref.extractall(mypath)
        zip_ref.close()
        ######## Zipping Done #########################
        temp_path = mypath + '/' + zip_file[:-4]
        temp_files = [f for f in listdir(temp_path) if isfile(join(temp_path, f))]
        start_index = len(batch_list)
        for image_file in temp_files:
            img_path = temp_path + '/' + image_file
            im = cv2.resize(cv2.imread(img_path), (448,448)).astype(np.float32)
    
            # Remove train image mean
            im[:,:,0] -= 103.939
            im[:,:,1] -= 116.779
            im[:,:,2] -= 123.68
            
            batch_list.append(im)
            batch_images.append(image_file[:-4])
            
        end_index = len(batch_list)
        batch_mappings[zip_file[:-4]] = [start_index,end_index]
        shutil.rmtree(temp_path)
	os.remove(mypath + '/' + zip_file)
    batch_array = np.array(batch_list)
    batch_features = model.predict(batch_array)
    final_features = get_maps(batch_features)
    keys = list(batch_mappings.keys())
    keys = sorted(keys,key=lambda x:int(x[x.find('_')+1:]))  ### Sorted index wise
    temp_array = []
    temp_images = []
    for b in keys:
        index_list = batch_mappings[b]
        feat_batch = final_features[index_list[0]:index_list[1]]
        img_batch = batch_images[index_list[0]:index_list[1]]
        batch_index = int(b[b.find('_')+1:]) - 1   #### batch_index to index the original_images_batches
        original_images = original_images_batches[batch_index] #### Should be 128
        for i in original_images:
            temp_index = img_batch.index(i)
            temp_images.append(i)
	    temp_array.append(feat_batch[temp_index])
    temp_array = np.array(temp_array)	  
    np.savez_compressed('train_img_maps/'+ keys[0] + keys[-1][keys[-1].find('_'):] +'.npz',temp_array)
    pickle.dump(temp_images,open('train_img_pickles/'+ keys[0] + '_' + keys[-1][keys[-1].find('_'):] + '_img_index.pkl','wb+'))
    
    
              
        
        



