# DYSLEX

Implementation of classifiers of the applied research project (Technology Agency of the Czech Republic, project No. [TL05000177](https://www.muni.cz/en/research/projects/61307)) on Diagnosis of dyslexia using eye-tracking and artificial intelligence. There are four different kinds of classifiers (kNN, MLP, ResNet18, ResNet50) trained for six different types of tasks (T1–T6).

## How to build and run

* Install the requirements specified in the [requirements.txt](requirements.txt) file.
* Training phase: run the [model_training.ipynb](model_training.ipynb) file to transform input data, train and store classification models, and analyze results.
* Classification phase: run the [classifier.py](classifier.py) file to perform classification on a specified subject and given trained models.

## Training phase

The [model_training.ipynb](model_training.ipynb) file contains methods to: (1) transform specific eye-tracking characteristics of each subject and each type of task into structured formats (both textual and *FixIma*-image formats) that are suitable for training the classifiers, (2) define the four types of classification models, (3) train the defined models using different hyperparameters, (4) analyze the classification results.

### Transformation of characteristics

For each type of task, the input eye-tracking data of each subject is transformed into a representation of:
* feature vector of real numbers with derived characteristics suitable for the given task, and
* *FixIma* image illustrating the position, size and duration of eye fixations using colored ellipses.

Sizes of feature vectors are 450, 5, 5, 34, 34, and 275 for tasks T1–T6, respectively (e.g., for the T1 task, 5 characteristics are chosen for each of the 90 considered syllables). The size of the FixIma images varies according to the type of task so that the image is tightly bounded by the area of interest of the given task content displayed on the screen. The specification of characteristics is defined in textual files in [meta](data/meta/) folder; the sizes of FixIma images are specified in [properties.json](properties.json).

### Definition and training of classifiers

Each of these two types of representations is classified by two different methods. The feature vector representation is classified using *k*-nearest neighbor (kNN) and Multilayer perceptron (MLP) approaches. The FixIma image representation
is classified by two variants of residual neural networks (ResNet18 and ResNet50). Given a specific task, each of these four classification models returns estimated probability of how much a given subject belongs to the dyslexic/intact class. The models are specified as follows.

* kNN – searching for the most similar subjects from the database based on comparison of feature vectors using the Euclidean distance. The final class is determined based on 
dyslexic/intact classes of the *k* most similar retrieved subjects. The kNN classifier enables specification of the parameter *k*.
```console
training_param_kNN_k_values = [1, 3, 4, 5, 10]
```

* MLP – a neural network with an input layer of the size of the feature vector, further one (or two) inner layer of the half size, and an output layer with size 2 (dyslexic/intact classes). In addition to the number of training iterations, the tested hyperparameters include *dropout* and *learning_rate*.
```console
training_param_MLP_layer_count_values = [1]
training_param_MLP_drop_values = [0.0, 0.2, 0.5]
training_param_MLP_epoch_count_values = [10, 20]
training_param_MLP_lr_values = [0.1, 0.05]
```

* ResNet18/50 — residual neural networks for classification of generated FixIma images. Both network variants are pretrained on a general image domain and fine-tuned on FixIma images by varying the number of training iterations and the *learning_rate* parameter.
```console
training_param_CNN_epoch_count_values = [20, 50]
training_param_CNN_lr_values = [0.001, 0.0001, 0.00001]
```

## Classification phase

The [classifier.py](classifier.py) file performs the classification of a given subject (or a set of subjects) on specified tasks and given trained models. The result is the estimated probability for each subject/task/model.
