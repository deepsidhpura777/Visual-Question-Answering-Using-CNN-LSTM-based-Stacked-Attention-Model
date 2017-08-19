def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq



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


img_path  = '/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/val_img_maps'
ques_path = '/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/val_questions'


img_onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
sort_key = lambda x:int(x[x.find('_')+1:x.rfind('_')])
img_onlyfiles = sorted(img_onlyfiles,key=sort_key)

f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()

max_answers = 3000

num_epochs = 3  ### Changed from 120 for testing
num_hidden_units = 1024
num_classes = max_answers
word_dim = 300
img_dim = 2048
max_len = 25 #### Time step sequence length


attention_model = nn_architectures.functional_network(num_classes,num_hidden_units,max_len,word_dim,img_dim)
attention_model.load_weights('hopeful_weights/epoch_99_weights.hdf5')
attention_model.compile(loss='categorical_crossentropy', optimizer='adam')
#attention_model = load_model('hopeful_weights/model_99.h5')


y_predict_all = []

time_step = 25

for i in img_onlyfiles: #### 35 files in all
	images = np.load(img_path + '/' + i)['arr_0']
        images = extract_features.normalize(images)
        questions = pickle.load(open(ques_path + '/' + i[:-4] + '_questions.pkl','rb+'))
        questions = extract_features.get_word_vectors(questions,len(questions),time_step,model)

        for j in range(0,len(questions),166):
            img_batch = images[j:j+166]
            ques_batch = questions[j:j+166]

            y_predict_batch = attention_model.predict([img_batch,ques_batch],batch_size=166)
            y_predict_all.append(list(y_predict_batch))
            del img_batch
            del ques_batch
                        

        del images
        del questions


f = open('final_predictions.pkl','wb+')
pickle.dump(y_predict_all,f)
f.close()






