import extract_features
import nn_architectures
import numpy as np
from random import shuffle
import time
import scipy.io as sc
import cPickle as pickle
from os import listdir
from os.path import isfile, join
from keras.models import load_model
import keras.backend as K

img_path  = '/home/rcf-proj2/jtr/dsidhpur/final_features/train_img_maps'
ques_path = '/home/rcf-proj2/jtr/dsidhpur/final_features/train_questions'
ans_path = '/home/rcf-proj2/jtr/dsidhpur/final_features/train_answers'


img_onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
sort_key = lambda x:int(x[x.find('_')+1:x.rfind('_')])
img_onlyfiles = sorted(img_onlyfiles,key=sort_key)
 

max_answers = 3000

num_epochs = 24  ### Changed from 120 for testing
num_hidden_units = 1024
num_classes = max_answers
word_dim = 300
img_dim = 2048
max_len = 15  #### Time step sequence length is 15 now !!!!
vocabulary = 12951
decay_factor = 0.99997592083
attention_model = nn_architectures.functional_embed_network(num_classes,num_hidden_units,max_len,word_dim,img_dim,vocabulary)

time_step = 15
step = 0
for e in range(num_epochs):
        now = time.time()
        for i in img_onlyfiles: #### ~~ 50 files in all
                images = np.load(img_path + '/' + i)['arr_0']
                images = extract_features.normalize(images)
             
                questions = np.load(ques_path + '/' + i[:-4] + '_questions.npz')['arr_0']
                answers = np.load(ans_path + '/' + i[:-4] + '_answers.npz')['arr_0']
	        ctr = 0	
                for j in range(0,len(images),128):
                        img_batch = images[j:j+128]
                        ques_batch = questions[ctr][:,:15]  ####### Changing the sequence size to new 15
                        ans_batch = answers[ctr]
			ctr += 1

                        loss = attention_model.train_on_batch([img_batch,ques_batch],ans_batch)
			step += 1
                        del img_batch
                        del ques_batch
                        del ans_batch
		        #old_lr = float(K.get_value(attention_model.optimizer.lr))
			#new_lr= 0.5 ** (step/50000.0) * old_lr
                        #K.set_value(attention_model.optimizer.lr, new_lr)
                del images
                del questions
                del answers


        print "Epoch Finished:",e," Loss:",loss
        later = time.time()
        print "Time per epoch in seconds:",int(later-now)
        print
	print "Steps:",step



attention_model.save('final_embed_weights/model_' + str(e) + '.h5')
