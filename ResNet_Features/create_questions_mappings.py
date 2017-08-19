def get_labels(answers): #### raw answers equivalent to the batch size  
    dum = pd.get_dummies(answers)
    matrix = np.array(dum,dtype=np.float64)
    return matrix

def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq


import cPickle as pickle
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


flatten = lambda l: [item for sublist in l for item in sublist]
mypath = '/home/rcf-proj2/jtr/dsidhpur/ResNet_Features/val_img_maps'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]    # batch_1_51.npz inclusive  ### The name of the files would give the batch numbers to combine !!

questions_batches = pickle.load(open('questions_val_batches.pkl','rb+'))
#answers_batches = pickle.load(open('answers_batches.pkl','rb+'))

#flatten_answers = flatten(answers_batches)
#answers = list(get_labels(flatten_answers))

#answers_batches = split_seq(answers,1798)


flatten = lambda l: [item for sublist in l for item in sublist]

for i in onlyfiles:
    indices = i[ i.find('_')+1 : i.find('.')].split('_')
    temp_q = questions_batches[int(indices[0])-1 : int(indices[1])]
    #temp_a = answers_batches[int(indices[0])-1 : int(indices[1])]
    q = flatten(temp_q)
    #a = flatten(temp_a)
    pickle.dump(q,open('val_questions' + '/' + i[:-4] + '_questions.pkl','wb+'))
    #pickle.dump(a,open('train_answers' + '/' + i[:-4] + '_answers.pkl','wb+'))






