import cPickle as pickle
import numpy as np

f = open('/home/rcf-proj2/jtr/dsidhpur/VQA/glove_vectors.pkl','rb+')
glove = pickle.load(f)
f.close()

f = open('original_word_index.pkl','rb+')
word_to_index = pickle.load(f)
f.close()

vocab = len(word_to_index.keys())
embedding_matrix = np.zeros((vocab+1,300))  #### Words not in gloves would be zero, +1 for end of sequences padding value !!

for i in word_to_index:
	if i in glove:
		embedding_matrix[word_to_index[i]] = glove[i]

f = open('embeddings.pkl','wb+')
pickle.dump(embedding_matrix,f)


