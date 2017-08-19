# Visual Question Answering using CNN-LSTM based Stacked Attention Model

Custom Neural Network architecture based on stacking image embeddings from CNNs (ResNet 152) and question embeddings from LSTMs to predict correct answers for context relevant questions about the images in VQA 1 validation dataset. The model accuracy is 55.7 % using the standard VQA evaluation metric, within 5 % range of the state-of-the-art model.


## Prerequisites

* Python 2.7 (Anaconda distribution favorable)
* Keras 1.1.1
* TensorFlow 0.9.0

## Getting Started

* Download the Training images from http://msvocds.blob.core.windows.net/coco2014/train2014.zip and validation images from http://msvocds.blob.core.windows.net/coco2014/val2014.zip

* Run create_batch_directories.py in new_data folder after changing the appropiate paths. Zip all the individual folders. There should be around 1798 zipped files.

* Run the keras_maps_extractor.py in the final_features folder to obtain the image maps.

* Run the embed_attention_vqa.py in Stacked_Attention_Models to obtain the model configurations after the first epoch.

* Run the continue_embed_vqa.py in Stacked_Attention_Models to execute the training procedure batch-wise. The training is to be done in batches due to the large size of the dataset. Each batch of 128 images has image maps of size approximately 3 GB and there are around 1798 batches !

* Run the embed_attention_predictions.py after loading the trained model and saving the weights in a separate file to obtain the prediction probabilities.

* Run accuracy_code.py in misc folder to obtain the final accuracy in terms of the VQA metric.

## Description of the files

### nn_architectures.py

Keras implementation of the stacked attention model. It includes multiple neural network architectures with different hyperparameters. The final architecture is defined as functional_embed_network.

### final_outputs

Includes the Neural Network training log describing the loss after each epoch. The best accuracy was at epoch 84, described in the accuracy_epoch_84

### Soft Attention Models

Includes pure TensorFlow as well as Keras implementation of the soft attention models using the VGG features and the ResNet 152 features extracted through transfer learning and concatenated with the LSTM features. The best accuracy of the soft attention model is 53.5 %

### Stacked Attention Models

Includes files for the stacked attention mechanism. embed_attention_vqa.py is the main file which handles the training procedure of the neural network. embed_attention_predictions.py handles the testing procedure by loading the saved model.

### final_features and ResNet_Features

Includes the keras_maps_extractor.py that extracts the image maps of dimension 14x14x2048, that is 196 attention distributions of 2048 dimension using the Keras implementation of the ResNet 152 architecture. The keras_val_extractor.py in ResNet_Features includes the code for feature extraction of soft attention mechanism


## Acknowledgments

* Vahid Kazemi, Ali Elqursh, Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering, arXiv:1704.03162, 2016
* James Chuanggg https://github.com/JamesChuanggg/
* Avi Singh https://avisingh599.github.io/deeplearning/visual-qa/
