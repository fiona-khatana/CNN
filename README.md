## The objective of this project
* Implement the forward and backward passes as well as the neural network training procedure
* Implement the widely-used optimizers and training tricks including dropout
* Get familiar with TensorFlow by training and designing a network on your own
* Learn how to fine-tune trained networks
* Visualize the learned weights and activation maps of a ConvNet
* Use Grad-CAM to visualize and reason why ConvNet makes certain predictions

## Work on the project
Working on the project in a virtual environment is highly encouraged.
Please see below for executing a virtual environment.
```shell
cd implement_cnn
sudo pip install virtualenv # If you didn't install it
virtualenv -p python3 /your/path/to/the/virtual/env
source  /your/path/to/the/virtual/env/bin/activate
pip install -r requirements.txt  # Install dependencies
# Note that this does NOT install TensorFlow,
# which you need to do yourself.
# Work on the project
deactivate # Exit the virtual environment
```

Please clone or download as .zip file of this repository.

## Work with IPython Notebook
To start working on the project, simply run the following command to start an ipython kernel.
```shell
# port is only needed if you want to work on more than one notebooks
jupyter notebook --port=/your/port/
```
and then work on each problem with their corresponding ```.ipynb``` notebooks.

In this project, please use Python `2.7`. You will need to make sure that your virtualenv setup is of the correct version of python is used. 

## Problems

### Problem 1: Basics of Neural Networks
The IPython Notebook `Problem_1.ipynb` will walk you through implementing the basic neural networks.

### Problem 2: Getting familiar with TensorFlow
The IPython Notebook `Problem_2.ipynb` will help you have a better understanding of implementing a simple ConvNet in Tensorflow.

### Problem 3: Training and Fine-tuning on Fashion MNIST and MNIST
The IPython Notebook `Problem_3.ipynb` will walk you through training a neural network from scratch on a datase and fine-tuning on another dataset.

### Problem 4: Visualizations and CAM
The IPython Notebook `Problem_4.ipynb` will help you have a better understanding of the skills of visualizing neural networks.
