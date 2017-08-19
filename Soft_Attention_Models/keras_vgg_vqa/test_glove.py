import extract_features
import nn_architectures
import numpy as np
from random import shuffle
from keras.utils import generic_utils
import gensim
import time
import cPickle as pickle
##### Need to do something about the wordtovec model that gives the word vectors !!!
questions_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
answers_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_train = open('/home/rcf-proj2/jtr/dsidhpur/VQA/data/images_train2014.txt', 'r').read().decode('utf8').splitlines()
max_answers = 1000

questions,answers,images = extract_features.select_k_best_examples(questions_train,answers_train,images_train,max_answers)

f = open('glove_vectors.pkl','rb+')
model = pickle.load(f)
f.close()
questions = extract_features.get_word_vectors(questions,len(questions),30,model)
questions = list(questions)
f = open('train_questions_gloves.pkl','wb+')
pickle.dump(questions,f)
print questions[0]

