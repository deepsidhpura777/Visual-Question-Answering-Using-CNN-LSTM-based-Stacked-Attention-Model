def func(batches):
    for i in range(len(batches)):
        if len(batches[i]) < 128:
            batches[i].append(batches[i][0])


import extract_features
import nn_architectures
import numpy as np
from random import shuffle
import time
import scipy.io as sc
import cPickle as pickle

test_file = '/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/trainImageMaps_resnet_pickles/dir_001.npz'
image_maps = extract_features.get_maps(test_file)  ### List of 196x2048 feature maps !
image_ids = pickle.load(open('/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/trainImageMaps_resnet_pickles/dir_001_img_index.pkl','rb+'))
train_questions_dic = pickle.load(open('/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/train_map.pkl','rb+'))
train_answers_dic = pickle.load(open('/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/answer_train_map.pkl','rb+'))
questions_list = []
answers_list = []
images_list = []
f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()

questions_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
answers_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_train2014.txt', 'r').read().decode('utf8').splitlines()
max_answers = 3000

_,__,images = extract_features.select_k_best_examples(questions_train,answers_train,images_train,max_answers)

new_image_ids = []
new_image_maps = []
for i in range(len(image_ids)):
    if str(image_ids[i]) in images:
	new_image_ids.append(image_ids[i])
	new_image_maps.append(image_maps[i])


for i in range(len(new_image_ids)):
    key = new_image_ids[i]
    questions_list += train_questions_dic[str(key)]
    answers_list += train_answers_dic[str(key)]
    cur_image_map = new_image_maps[i]
    for j in range(len(train_questions_dic[str(key)])):
	images_list.append(cur_image_map) 

#questions,answers,images = extract_features.select_k_best_examples(questions_list,answers_list,images_maps,max_answers)

answers = extract_features.get_labels(answers_list)
answers = list(answers)

questions = extract_features.get_word_vectors(questions_list,len(questions_list),25,model)
questions = list(questions)

len_q = len(questions)

print "questions len before split:",len(questions)

questions_batches = extract_features.split_seq(questions,24) ### Each batch will be appx  1798~~128.
answers_batches = extract_features.split_seq(answers,24)
images_batches = extract_features.split_seq(images_list,24)

#func(questions_batches)
#func(answers_batches)
#func(images_batches)


num_epochs = 4  ### Changed from 120 for testing
num_hidden_units = 1024
#num_classes = max_ansewrs
num_classes = 800
word_dim = 300
img_dim = 2048
max_len = 25 #### Time step sequence length

attention_model = nn_architectures.attention_network(num_classes,num_hidden_units,max_len,word_dim,img_dim) 

json_string = attention_model.to_json()
model_file_name = 'test_attention'
open(model_file_name  + '.json', 'w').write(json_string)

model_save_interval = 20
time_step = 25

print "Model Started training, Bitch !!!"

for i in range(num_epochs):

    now = time.time()
    for j in range(len(questions_batches)):

        X_q_batch = np.array(questions_batches[j])
        X_i_batch = np.array(images_batches[j])
        y_batch = np.array(answers_batches[j])
	#print "Q batch shapes:",X_q_batch.shape
	#print "I batch shapes:",X_i_batch.shape
	#print "Y batch shape:",y_batch.shape
        #loss = attention_model.train_on_batch([[[X_i_batch,X_q_batch],X_i_batch],X_q_batch],y_batch)
	loss = attention_model.train_on_batch([X_i_batch,X_q_batch],y_batch)
    print "Batch No:",j,"Training Loss",loss
    print
    print "Epoch Finished:",i
    later = time.time()
    print "Time per epoch in seconds:",int(later-now)
    if i % model_save_interval == 0:
        attention_model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(i))

attention_model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(i+1))
