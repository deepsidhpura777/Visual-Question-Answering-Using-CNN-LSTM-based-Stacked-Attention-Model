
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Reshape,Merge
from keras.layers.recurrent import LSTM
from multi_gpu import make_parallel
from keras.optimizers import Adam
from keras.layers import Embedding
from sklearn.preprocessing import normalize
from keras.layers import Conv1D,Flatten,Input,merge
import tensorflow as tf
from multi_gpu import make_parallel

def weighted_average(outputs): ### takes a list of inputs from the attention model and the input model
    attention_outputs = outputs[0]
    image_outputs = outputs[1]
    
    weights_1 = attention_outputs[:,:,0]
    weights_1 = weights_1[:,:int(weights_1.get_shape()[1])-1]
    weights_2 = attention_outputs[:,:,1] 
    weights_2 = weights_2[:,:int(weights_2.get_shape()[1])-1]
    #print 'weights_1_shape',weights_1.get_shape()
    #print 'weights_2_shape',weights_2.get_shape()
    l1 = []
    l2 = []

    for i in range(166): ### changed to 166 for testing !!!!! repeats batch size number of times
        t1 = weights_1[i,:]
        t2 = weights_2[i,:]
        t1 = t1 / tf.reduce_sum(t1)
        t2 = t2 / tf.reduce_sum(t2)
	temp_w_1 = tf.matmul(tf.reshape(t1,(int(t1.get_shape()[0]),1)),tf.ones((1,int(image_outputs.get_shape()[2]))))
        #temp_w_1 = tf.reshape(t1,(len(t1),1)).dot(np.ones((1,image_outputs.shape[2])))
	temp_w_2 = tf.matmul(tf.reshape(t2,(int(t2.get_shape()[0]),1)),tf.ones((1,int(image_outputs.get_shape()[2]))))
        #temp_w_2 = tf.reshape(t2,(len(t2),1)).dot(np.ones((1,image_outputs.shape[2])))
        temp_i = image_outputs[i,:,:]

        x1 = tf.mul(temp_w_1,temp_i)
        x2 = tf.mul(temp_w_2,temp_i)
        
        l1.append(x1)
        l2.append(x2)
    l1 = tf.convert_to_tensor(l1)  #### (None,392,1024)
    l2 = tf.convert_to_tensor(l2) #### (None,392,1024)
    
    l1 = tf.reduce_sum(l1,reduction_indices=[1]) ##### (None,1024)
    l2 = tf.reduce_sum(l2,reduction_indices=[1]) ##### (None,1024)

    #print 'l1_shape',l1.get_shape() 
    #print 'l2_shape',l2.get_shape()       

    l = tf.concat(1,[l1,l2]) ##### (None,2048)
    #print 'l_shape',l.get_shape()
    
    return l

def custom_reshape_merge(outputs):
    avg = outputs[0]
    lstm = outputs[1]
    #print 'lstm_shape before',lstm.get_shape()
    lstm = lstm[:,0,:]
    #print 'lstm_shape',lstm.get_shape()
    comb = tf.concat(1,[avg,lstm])

    #print 'comb_shape',comb.get_shape()
    return comb


def functional_network(num_classes,num_hidden_units,max_len,word_dim,img_dim):
    
    image_input = Input(shape=(196,img_dim))
    image_output = Reshape((196*(img_dim/num_hidden_units),num_hidden_units))(image_input)
    
    lstm_input = Input(shape=(max_len,word_dim))
    lstm = Dropout(0.5)(lstm_input)
    lstm = LSTM(num_hidden_units,activation='tanh',return_sequences=False)(lstm_input)
    lstm_output = Reshape((1,num_hidden_units))(lstm)
    
    attention = merge([image_output,lstm_output],mode='concat',concat_axis=1)
    attention = Dropout(0.5)(attention)
    attention = Conv1D(512,1)(attention)
    attention = Activation('relu')(attention)
    attention = Dropout(0.5)(attention)
    attention = Conv1D(2,1)(attention)
    attention_output = Activation('softmax')(attention)
    
    average = merge([attention_output,image_output],mode=weighted_average,output_shape=(2048,))
    
    combined = merge([average,lstm_output],mode=custom_reshape_merge,output_shape=(3072,))
    combined = Dropout(0.5)(combined)
    combined = Dense(1024,init='glorot_uniform')(combined)
    combined = Activation('relu')(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(num_classes,init='glorot_uniform')(combined)
    combined_output = Activation('softmax')(combined)
    
    
    model = Model(input=[image_input,lstm_input],output=combined_output) #### Not sure if list is to be passed for outputs. No list necessary !!
    adam = Adam(decay=0.000002)
    print 'Model Compilation started'
    model.compile(loss='categorical_crossentropy',optimizer=adam) #### adam,rmsprop better than sgd for rnns
    print 'Model compilation done' 
    
    return model




def functional_embed_network(num_classes,num_hidden_units,max_len,word_dim,img_dim,vocabulary):
    
    image_input = Input(shape=(196,img_dim))
    image_output = Reshape((196*(img_dim/num_hidden_units),num_hidden_units))(image_input)

    lstm_input = Input(shape=(max_len,))
    embedding_input = Embedding(vocabulary + 1,word_dim,init='glorot_uniform',input_length=max_len)(lstm_input)
    embedding_input = Activation('tanh')(embedding_input)
    lstm = Dropout(0.5)(embedding_input)
    lstm = LSTM(num_hidden_units,activation='tanh',return_sequences=False)(lstm)
    lstm_output = Reshape((1,num_hidden_units))(lstm)

    attention = merge([image_output,lstm_output],mode='concat',concat_axis=1)
    attention = Dropout(0.5)(attention)
    attention = Conv1D(512,1)(attention)
    attention = Activation('relu')(attention)
    attention = Dropout(0.5)(attention)
    attention = Conv1D(2,1)(attention)
    attention_output = Activation('softmax')(attention)

    average = merge([attention_output,image_output],mode=weighted_average,output_shape=(2048,))

    combined = merge([average,lstm_output],mode=custom_reshape_merge,output_shape=(3072,))
    combined = Dropout(0.5)(combined)
    combined = Dense(1024,init='glorot_uniform')(combined)
    combined = Activation('relu')(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(num_classes,init='glorot_uniform')(combined)
    combined_output = Activation('softmax')(combined)


    model = Model(input=[image_input,lstm_input],output=combined_output) #### Not sure if list is to be passed for outputs. No list necessary !!
    #adam = Adam(decay=0.00002)
    print 'Model Compilation started'
    model.compile(loss='categorical_crossentropy',optimizer='adam') #### adam,rmsprop better than sgd for rnns
    print 'Model compilation done'

    return model




def attention_network(num_classes,num_hidden_units,max_len,word_dim,img_dim): #### img dim here is the depth, ie 2048 (14x14x2048)
    image_model = Sequential()
    image_model.add(Reshape((196,img_dim),input_shape = (196,img_dim),name='Image_Reshape_1',trainable=False))
    image_model.add(Reshape((196*(img_dim/num_hidden_units),num_hidden_units),name='Image_Reshape_2',trainable=False)) ##### (None,392,1024)

    lstm_model = Sequential()
    lstm_model.add(LSTM(num_hidden_units,activation='tanh',return_sequences=False,input_shape = (max_len,word_dim),name='LSTM_1',trainable=True))
    lstm_model.add(Reshape((1,num_hidden_units),name='LSTM_Reshape_1',trainable=False))
      
    attention_model = Sequential()
    attention_model.add(Merge([image_model,lstm_model],mode='concat',concat_axis=1,name='Merge_1'))
    attention_model.add(Dropout(0.5,trainable=False))
    attention_model.add(Conv1D(512,1,name='Conv_1',trainable=True))
    attention_model.add(Activation('relu',trainable=False))
    attention_model.add(Conv1D(2,1,name='Conv_2',trainable=True))
    attention_model.add(Activation('softmax',trainable=False))
    
    average_model = Sequential()
    average_model.add(Merge([attention_model,image_model],mode=weighted_average,output_shape=(2048,),name='Merge_2'))
    
    combined_model = Sequential()
    combined_model.add(Merge([average_model,lstm_model],mode=custom_reshape_merge,output_shape=(3072,),name='Merge_3'))
    combined_model.add(Dropout(0.5,trainable=False))
    combined_model.add(Dense(1024,init='glorot_uniform',name='Dense_1',trainable=True))
    combined_model.add(Activation('relu',trainable=False))

    combined_model.add(Dropout(0.5,trainable=False))
    combined_model.add(Dense(num_classes,init='glorot_uniform',name='Dense_2',trainable=True))
    combined_model.add(Activation('softmax',trainable=False))
    
    adam = Adam(decay=0.00002)
    print 'Model Compilation started'
    combined_model.compile(loss='categorical_crossentropy',optimizer=adam) #### adam,rmsprop better than sgd for rnns
    print 'Model compilation done'
    
    
    #### Add sklearn l1 normalize to the output of the attention_model except the last one.
    #### There are 2 attention distributions of dimension 393, which act as weights for each of the 392 features
    #### Multiply it with 392 features each of dimension 1024
    #### Take the sum of these 392 features after multiplication across axis=0
    #### Concatenate 2 1024 dimensional vectors and proceed as normal !
    

    return combined_model    



def lstm_resnet_network(num_classes,num_hidden_units,max_len,word_dim,img_dim):
    image_model = Sequential()
    image_model.add(Reshape((img_dim,),input_shape = (img_dim,)))
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(num_hidden_units,activation='tanh',return_sequences=False,input_shape = (max_len,word_dim)))
    lstm_model.add(Dropout(0.5))

	

    combined_model = Sequential()
    combined_model.add(Merge([image_model,lstm_model],mode='concat',concat_axis=1))
    combined_model.add(Dense(1024,init='glorot_uniform'))
    combined_model.add(Activation('relu'))
    
    combined_model.add(Dropout(0.5))
    combined_model.add(Dense(num_classes,init='glorot_uniform'))
    combined_model.add(Activation('softmax'))
    
    print 'Model Compilation started'
    combined_model.compile(loss='categorical_crossentropy',optimizer='adam') #### adam,rmsprop better than sgd for rnns
    print 'Model compilation done'
    return combined_model


def lstm_resnet_embedding_network(num_classes,num_hidden_units,max_len,word_dim,img_dim,embedding_matrix):
    image_model = Sequential()
    image_model.add(Reshape((img_dim,),input_shape = (img_dim,)))

    lstm_model = Sequential()
    lstm_model.add(Embedding(embedding_matrix.shape[0],word_dim,weights=[embedding_matrix],trainable=True,input_length=max_len))
    lstm_model.add(Activation('tanh'))
    lstm_model.add(LSTM(num_hidden_units,activation='tanh',return_sequences=False,input_shape = (max_len,word_dim)))
    lstm_model.add(Dropout(0.5))

    combined_model = Sequential()
    combined_model.add(Merge([image_model,lstm_model],mode='concat',concat_axis=1))
    combined_model.add(Dense(1024,init='glorot_uniform'))
    combined_model.add(Activation('relu'))
    combined_model.add(Dropout(0.5))
    combined_model.add(Dense(num_classes,init='glorot_uniform'))
    combined_model.add(Activation('softmax'))
    #adam = Adam(decay=0.99997592083)
    print 'Model Compilation started'
    combined_model.compile(loss='categorical_crossentropy',optimizer='adam') #### adam,rmsprop better than sgd for rnns
    print 'Model compilation done'
    return combined_model
