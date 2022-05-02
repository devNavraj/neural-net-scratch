from neural_net import NeuralNetwork
from load_mnist_data import CustomDataLoader
from settings import *


if __name__ == '__main__':

    print(
        '''
        Loading the training and testing data from the CustomDataLoader.
        '''
    )
    mnist_loader = CustomDataLoader()
    train_data, test_data = mnist_loader.preprocess_data()

    print(
        '''
        ------------------------------------------------------------------------
        ------------------------------------------------------------------------
        ------------------------------------------------------------------------
        Starting training for neural network of three layers (one hidden layer).
        ------------------------------------------------------------------------
        ------------------------------------------------------------------------
        ------------------------------------------------------------------------
        '''
    )
    simple_nn = NeuralNetwork(THREE_LAYER_NN, activation='sigmoid')
    simple_nn.mini_batch_gradient_descent(
        train_data=train_data, 
        mini_batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        learning_rate=ETA, 
        test_data=test_data
    )

    print(
        '''
        Again unpacking the training and testing data from the CustomDataLoader.
        '''
    )
    train_data, test_data = mnist_loader.preprocess_data()

    print(
        '''
        --------------------------------------------------------------------------
        --------------------------------------------------------------------------
        --------------------------------------------------------------------------
        Starting training for neural network of five layers (three hidden layers).
        --------------------------------------------------------------------------
        --------------------------------------------------------------------------
        --------------------------------------------------------------------------
        '''
    )
    five_layer_nn = NeuralNetwork(THREE_HIDDEN_LAYER_NN, activation='sigmoid')
    five_layer_nn.mini_batch_gradient_descent(
        train_data=train_data, 
        mini_batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        learning_rate=ETA, 
        test_data=test_data
    )