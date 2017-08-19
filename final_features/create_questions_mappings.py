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
mypath = '/home/rcf-proj2/jtr/dsidhpur/final_features/val_img_maps'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]    # batch_1_51.npz inclusive  ### The name of the files would give the batch numbers to combine !!

questions_batches = np.load('questions_val_batches_array.npz')['arr_0']
#answers_batches = np.load('answers_batches_array.npz')['arr_0']


flatten = lambda l: [item for sublist in l for item in sublist]

for i in onlyfiles:
    indices = i[ i.find('_')+1 : i.find('.')].split('_')
    temp_q = questions_batches[int(indices[0])-1 : int(indices[1])]
    #temp_a = answers_batches[int(indices[0])-1 : int(indices[1])]
  
    np.savez_compressed('val_questions' + '/' + i[:-4] + '_questions.npz',temp_q)
    #np.savez_compressed('train_answers' + '/' + i[:-4] + '_answers.npz',temp_a)






