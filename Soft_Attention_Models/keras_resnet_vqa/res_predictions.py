import extract_features
import nn_architectures
import numpy as np
from keras.utils import generic_utils
import gensim
from keras.models import model_from_json
import cPickle as pickle
import tensorflow as tf
from multi_gpu import make_parallel
import scipy.io as sc

questions_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
answers_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_val2014_all.txt', 'r').read().decode('utf8').splitlines()
images_val = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_val2014_all.txt', 'r').read().decode('utf8').splitlines()


f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()

f = open('/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/resnet_val_features.pkl')
cnn_model = pickle.load(f)
f.close()

answers_val = map(lambda x:x.split(';'),answers_val)

questions_val = extract_features.get_word_vectors(questions_val,len(questions_val),25,model)
questions_val = list(questions_val)
questions_val_batches = extract_features.split_seq(questions_val,83)

vectors_needed = images_val[:]

images = extract_features.get_cnn_vectors(vectors_needed,cnn_model)
images = list(images)

images_val_batches = extract_features.split_seq(images,83)

model = model_from_json(open("/home/rcf-proj2/jtr/dsidhpur/VQA/res_vqa/lstm_resnet_different.json").read())
model.load_weights("/home/rcf-proj2/jtr/dsidhpur/VQA/res_vqa/lstm_resnet_different_epoch_100.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

y_predict_all = []

for i in range(len(questions_val_batches)):
    X_q_val_batch = np.array(questions_val_batches[i])
    X_i_val_batch = np.array(images_val_batches[i])
    y_predict_batch = model.predict([X_i_val_batch,X_q_val_batch], verbose=1)
    y_predict_all.append(list(y_predict_batch))

f = open('res_different_100_predictions.pkl','wb+')
pickle.dump(y_predict_all,f)
