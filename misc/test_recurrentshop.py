#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 18:30:23 2017

@author: deepsidhpura
"""

from recurrentshop import *
from keras.layers import *
from keras.models import Model

rnn = RecurrentSequential(return_states=True)
l1 = LSTMCell(512,input_dim=300)
rnn.add(l1)
l2 = LSTMCell(512,input_dim=512)
rnn.add(l2)

a = Input((864,300))
b = rnn(a)
model = Model(a, b)
model.summary()

