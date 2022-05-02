import numpy as np
from utils import sigmoid, altered_sigmoid, mse_cost_function
from settings import SEED


class NeuralNetwork:
    '''
    This is a custom neural netwok package built from 
    scratch with numpy.
    '''
    
    def __init__(self, sizes, activation='sigmoid', cost_function='mse', seed=42):
        '''
        Instantiate the weights and biases of the network.
        ---
        Params:
        ---
        sizes: a list of the number of neurons in each layer
        activation: function that handles non-linearity of 
        output from each dense layer, default value is sigmoid.
        '''
        self.sizes = sizes
        self.num_layers = len(self.sizes)
        self.seed = seed if seed is not None else SEED

        np.random.seed(self.seed)

        # Choose activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
        elif activation == 'alternative':
            self.activation = altered_sigmoid
        else: 
            raise ValueError(
                '''
                Activation function is currently not supported, 
                please use 'sigmoid' or 'alternative' instead.
                '''
            )

        if cost_function == 'mse':
            self.cost_function = mse_cost_function

        # Save all weights and biases
        self.weights = [np.random.randn(y, x)*np.sqrt(1/x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]


    def feed_forward(self, a):
        '''
        Perform a feed forward computation in the neural network.
        0---
        Params:
        ---
        a: data to be fed to the network with
        shape: (input_shape, batch_size). 
        Also represents the current activation vector.
        ---
        Returns:
        ---
        a: ouptut activation (output_shape, batch_size)
        '''
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activation(z)
        return a


    def back_propagate(self, x, y):
        '''
        Applies back-propagation and computes the gradient of the
        cost function w.r.t the weights and biases of the network.
        ---
        Params:
        ---
        x: list of deltas computed by compute_deltas
        y: true label vector (real target) from each batch
        ---
        Returns:
        ---
        del_weight: list of gradients w.r.t. the weight matrices of the network
        del_bias: list of gradients w.r.t. the biases (vectors) of the network
        '''
        del_weight = [np.zeros(w.shape) for w in self.weights]
        del_bias = [np.zeros(b.shape) for b in self.biases]

        # Feed-forward
        activation = x
        # List of activations per layer
        activations = [x]
        # List of pre-activations (z values) per layer
        pre_activations = [] 

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            pre_activations.append(z)
            activation = self.activation(z)
            activations.append(activation)
       
        # Back-propagation
        delta = self.cost_function(
            y, 
            activations[-1],
            derivative=True
        ) * self.activation(
                pre_activations[-1], 
                derivative=True
            )
        del_bias[-1] = delta
        del_weight[-1] = np.dot(delta, activations[-2].T)
       
        for layer in range(2, self.num_layers):
            z = pre_activations[-layer]
            activation_derivative = self.activation(z, derivative=True)
            delta = np.dot(self.weights[-layer+1].T, delta) * activation_derivative
            del_bias[-layer] = delta
            del_weight[-layer] = np.dot(delta, activations[-layer-1].T)
        return (del_weight, del_bias)

    
    def update_mini_batch(self, mini_batch, learning_rate, train_losses, train_accuracies):
        '''
        Update weights and biases for each mini-batch of training data.
        ---
        Params:
        ---
        mini_batch: batch of fixed number of training data (X_train, Y_train)
                    for each epoch
        learning_rate: value of learning rate of the model
        train_losses: list of loss(cost) values of the cost function for each 
                      iteration in mini_batch
        train_accuracies: list of accuracy values of the model for each 
                          iteration in mini_batch
        ---
        Returns:
        ---
        None
        '''
        del_weight = [np.zeros(w.shape) for w in self.weights]
        del_bias = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            dW, dB = self.back_propagate(x, y)
            for i, (dW_i, dB_i) in enumerate(zip(dW, dB)):
                del_weight[i] += dW_i / len(mini_batch)
                del_bias[i] += dB_i / len(mini_batch)
            
            # Computing the predicted value (y_pred) for each X_train values 
            y_train_pred = self.predict(x)
            # Computing the loss(cost) for the predicted value
            train_loss = self.cost_function(y, y_train_pred)
            train_losses.append(train_loss)

            # Computing the accuracy score for the predicted value
            train_accuracy = self.accuracy_score(y, y_train_pred)
            train_accuracies.append(train_accuracy)

        # Updating the weights and biases for the mini_batch
        self.weights = [w - learning_rate*dW for w, dW in zip(self.weights, del_weight)]
        self.biases = [b - learning_rate*dB for b, dB in zip(self.biases, del_bias)]


    def mini_batch_gradient_descent(self, train_data, mini_batch_size, epochs, learning_rate, test_data=None):
        '''
        Trains the network using the gradients computed by back-propagation
        Processes the training data by mini_batches and trains the network using 
        mini_batch gradient descent.
        It is a mixture of Batch Gradient Descent and Stochastic Gradient Descent.
        ---
        Params:
        ---
        train_data: input data that is used for training
        mini_batch_size: number of data points to process in each batch
        epochs: number of epochs (number of times we run through 
                the whole dataset) for the training
        learning_rate: value of the learning rate of the model
        test_data: input data that is used for testing
        ---
        Returns:
        ---
        history: dictionary of train and test metrics per each epoch
            train_loss: train loss
            train_accuracy: train accuracy
            test_accuracy: test accuracy
        '''
        history_train_losses = []
        history_train_accuracies = []
        history_test_accuracies = []

        train_losses=[]
        train_accuracies=[]

        train_data = list(train_data)
        sample_num = len(train_data)
       
        if test_data:
            test_data = list(test_data)
       
        for e in range(epochs):
            # Shuffling the data to ensure randomness
            np.random.shuffle(train_data)

            # Creating a list of mini_batches
            mini_batches = [train_data[k:k+mini_batch_size]
                            for k in range(0, sample_num, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, 
                    learning_rate, 
                    train_losses, 
                    train_accuracies
                )

            history_train_losses.append(np.mean(train_losses))
            history_train_accuracies.append(np.mean(train_accuracies))

            if test_data:
                test_accuracy = float(self.evaluate(test_data))
                history_test_accuracies.append(test_accuracy)
                print('Epoch {} / {} | train_loss: {} | train_accuracy: {} | test_accuracy : {} '.format(
                    e, epochs, np.round(np.mean(train_losses), 4), np.round(np.mean(train_accuracies), 4), 
                    np.round(test_accuracy, 4)))
            else:
                print('Epoch {} / {} | train_loss: {} | train_accuracy: {}'.format(
                    e, epochs, np.round(np.mean(train_losses), 4), 
                    np.round(np.mean(train_accuracies), 4)))

        print('Training completed successfully.')

        history = {
                'train_loss': history_train_losses, 
                'train_accuracy': history_train_accuracies,
                'test_accuracy': history_test_accuracies
            }

        return history


    def predict(self, a):
        '''
        Use the current state of the network to make predictions.
        ---
        Params:
        ---
        a: data to be fed to the network with
           shape: (input_shape, batch_size)
           Current activation vector
        ---
        Returns:
        ---
        predictions: vector of output predictions
        '''
        predictions = (self.feed_forward(a) > 0.5).astype(int)
        return predictions


    def accuracy_score(self, y_true, y_pred):
        '''
        Compute accuracy score for training data.
        ---
        Params:
        ---
        y_true: true label vector (real target)
        y_pred: prediction vector (output of the current activation function)
        ---
        Returns:
        ---
        accuracy score
        '''
        return np.sum(np.equal(y_true, y_pred)) / len(y_true)


    def evaluate(self, test_data):
        '''
        Evaluate the model on test data computing the accuracy 
        i.e. total no. of correct predictions / total no. of testing data
        ---
        Params:
        ---
        test_data: input data that is used for testing
        ---
        Returns:
        ---
        accuracy
        '''
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        accuracy = (sum(int(y[x]) for (x, y) in test_results) / len(test_data))
        return accuracy
            