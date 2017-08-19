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


img_path  = '/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/train_img_maps'
ques_path = '/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/train_questions'
ans_path = '/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/train_answers'


img_onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
sort_key = lambda x:int(x[x.find('_')+1:x.rfind('_')])
img_onlyfiles = sorted(img_onlyfiles,key=sort_key)

f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()

f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/res_vqa/labels_3000.pkl','rb+')
labels = pickle.load(f)
f.close()

max_answers = 3000

num_epochs = 3  ### Changed from 120 for testing
num_hidden_units = 1024
num_classes = max_answers
word_dim = 300
img_dim = 2048
max_len = 25 #### Time step sequence length


attention_model = load_model('hopeful_weights/model_76.h5')

time_step = 25
ctr = 0
for e in range(77,100):   ###### From epochs 77 to 99 inclusive  !!
        now = time.time()
        for i in img_onlyfiles: #### 35 files in all
                images = np.load(img_path + '/' + i)['arr_0']
                images = extract_features.normalize(images)
                questions = pickle.load(open(ques_path + '/' + i[:-4] + '_questions.pkl','rb+'))
                answers = pickle.load(open(ans_path + '/' + i[:-4] + '_answers.pkl','rb+'))

                answers = extract_features.extract_labels(answers,labels)  ### matrix of ~~ 6528 x 3000 size
                questions = extract_features.get_word_vectors(questions,len(questions),time_step,model)

                for j in range(0,len(questions),128):
                        img_batch = images[j:j+128]
                        ques_batch = questions[j:j+128]
                        ans_batch = answers[j:j+128]

                        loss = attention_model.train_on_batch([img_batch,ques_batch],ans_batch)
                        del img_batch
                        del ques_batch
                        del ans_batch
			
                del images
                del questions
                del answers
	
        print "Epoch Finished:",e," Loss:",loss
        later = time.time()
        print "Time per epoch in seconds:",int(later-now)
	print 



attention_model.save('hopeful_weights/model_' + str(e) +'.h5')
