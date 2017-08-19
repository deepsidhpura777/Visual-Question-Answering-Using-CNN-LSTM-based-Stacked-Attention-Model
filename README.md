# Visual Question Answering using CNN-LSTM based Stacked Attention Model

Custom Neural Network architecture based on stacking image embeddings from CNNs (ResNet 152) and question embeddings from LSTMs to predict correct answers for context relevant questions about the images in VQA 1 validation dataset. The model accuracy is 55.7 % using the standard VQA evaluation metric, within 5 % range of the state-of-the-art model.

## Description of the files

### nn_architectures.py

Keras implementation of the stacked attention model. It includes multiple neural network architectures with different hyperparameters. The final architecture is defined as functional_embed_network.

### Soft Attention Models

Includes pure TensorFlow as well as Keras implementation of the soft attention models using the VGG features and the ResNet 152 features extracted through transfer learning and concatenated with the LSTM features. The best accuracy of the soft attention model is 53.5 %

### Stacked Attention Models

Includes files for the stacked attention mechanism. embed_attention_vqa.py is the main file which handles the training procedure of the neural network. embed_attention_predictions.py handles the testing procedure by loading the saved model.

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
