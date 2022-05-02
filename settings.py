import os

# MNIST dataset directory path
ROOT_PATH = os.getcwd() 
DATA_DIR = 'data'
DATA_DIR_PATH = os.path.join(ROOT_PATH, DATA_DIR)

# Dataset names with extension
TRAIN_DATA = 'mnist_train.csv'
TEST_DATA = 'mnist_test.csv'

# Image Size
IMAGE_SIZE = 28

# Image Pixel Scale Factor
SCALE_FACTOR = 1.0 / 255

# Number of different labels
LABEL_NUM = 10

# Seed number
SEED = 42

# Hyperparameters
BATCH_SIZE = 10  # Mini-batch size
EPOCHS = 10  # Number of epochs
ETA = 0.04  # Learning rate

# List of number of neurons per layer in our neural network
# Input neuron size must be pixel_size = IMAGE_SIZE*IMAGE_SIZE 
# Output neuron size must be the number of labels (target_value)
THREE_LAYER_NN = [784, 64, 10] # Three layers (one input, one hidden, one output)
THREE_HIDDEN_LAYER_NN = [784, 128, 64, 32, 10] 