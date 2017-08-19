import extract_features
import nn_architectures
import numpy as np
from random import shuffle
import time
import scipy.io as sc
import cPickle as pickle
from os import listdir
from os.path import isfile, join
from keras.models import model_from_json
from keras.models import load_model


img_path  = '/home/rcf-proj2/jtr/dsidhpur/final_features/val_img_maps'
ques_path = '/home/rcf-proj2/jtr/dsidhpur/final_features/val_questions'


img_onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
sort_key = lambda x:int(x[x.find('_')+1:x.rfind('_')])
img_onlyfiles = sorted(img_onlyfiles,key=sort_key)

max_answers = 3000

num_epochs = 1  ### Changed from 120 for testing
num_hidden_units = 1024
num_classes = max_answers
word_dim = 300
img_dim = 2048
max_len = 15 #### Time step sequence length is 26 now !!!!
vocabulary = 12951

attention_model = nn_architectures.functional_embed_network(num_classes,num_hidden_units,max_len,word_dim,img_dim,vocabulary)

time_step = 15

attention_model.load_weights('final_embed_weights/epoch_82_weights.hdf5')
attention_model.compile(loss='categorical_crossentropy', optimizer='adam')

y_predict_all = []

for i in img_onlyfiles: #### 35 files in all
        images = np.load(img_path + '/' + i)['arr_0']
        images = extract_features.normalize(images)
        
	questions = np.load(ques_path + '/' + i[:-4] + '_questions.npz')['arr_0']
        ctr = 0
        for j in range(0,len(images),166):
            img_batch = images[j:j+166]
            ques_batch = questions[ctr][:,:15]
	    ctr += 1

            y_predict_batch = attention_model.predict([img_batch,ques_batch],batch_size=166)
            y_predict_all.append(list(y_predict_batch))
            del img_batch
            del ques_batch


        del images
	del questions




f = open('final_embed_predictions_110.pkl','wb+')
pickle.dump(y_predict_all,f)
f.close()
