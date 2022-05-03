# Neural Network from Scratch
I have built a simple Neural Network from scratch using Python and its NumPy library only. I have tried to implement this model for famous 'HelloWorld of Computer Vision' - MNIST Handwritten Digits Recognition Task. 
[GitHub Repo](https://github.com/devNavraj/neural-net-scratch)

## Dataset
[MNIST Dataset](http://yann.lecun.com/exdb/mnist/) contains 70,000 images of hand-written digits, 60,000 for training while 10,000 for testing, each 28×28 pixels, in grayscale with pixel-values from 0 to 255.
You can download this dataset from [Kaggle](https://www.kaggle.com/competitions/mnist-en/data).

The dataset contains one label for each image that specifies the digit that I see in each image. I say that there are 10 classes because I have 10 labels. I have downloaded the dataset and kept it on the **Data** folder.

## Requirements
To run this project, you need to have Python >= 3.7 and its NumPy library. To download NumPy you can run the following command in the terminal:

```bash
pip install numpy
```

## How to run?
1. Open this folder on your Code Editor or IDE (Visual Code, Atom, Sublime, PyCharm, or any)
2. Open the terminal and run the following command:
```bash
python main.py
```

### Overview of File Structure
```
.
├── README.md  
├── data
│   ├── mnist_test.csv
│   └── mnist_train.csv
├── load_mnist_data.py
├── main.py
├── neural_net.py
├── settings.py
└── utils.py
```

There are mainly five python files made here. 
- The first and the main file is *neural_net.py* file which will be outlined in “Setting Up Helper Functions” and “Building the Neural Network from Scratch”. 
- The second one is *load_mnist_data.py* file to load the training and testing data for the model, outlined in “Loading MNIST Data”. 
- The third is *utils.py* file that contains all helper functions that we need to import into neural_net module. These helper functions are actually activation functions and cost functions along with their derivatives that I have used to build the neural network model.
- The fourth one is *settings.py* which basically contains all the constant variables viz. Dataset Files Path, Image Size, Number of Labels (Classes), List of Number of Neurons in each Layer.
- Finally, I have created a file to test my neural network called *main.py* that can be run in the terminal. This file is outlined in “Running Tests”.

## Model Architecture
In machine learning terminology, each input to the neuron ${(x_1, x_2, ..., x_n)}$ is known as a feature, and each feature is weighted with a number to represent the strength of that input ${(w_{1_j}, w_{2_j}, ..., w_{n_j})}$. The weighted sum of inputs is then passes through an *activation function*, whose general purpose is to model the *firing rate* of a biological neuron by converting the weighted sum into a new number according to a formula. 

First, I have built a basic neural network with 3 layers: 1 input layer, 1 hidden layer and 1 output layer. Then, I have added two more hidden layers to it such that the total number of layers now become 5 from 3. All layers are fully connected i.e. dense. I have implemented popular Sigmoid activation function and a modified sigmoid function with positive *z* to the exponent part. 

Let's try to define the layers in an exact way. To be able to classify digits, you must end up with the probabilities of an image belonging to a certain class after running the neural network because then you can quantify how well your neural network performed.

### Simple Three Layer Model
This type of base neural network architecture is simply a Logistic Regression which is a popular traditional Machine Learning Classifier.
1. **Input Layer:** In this layer, I input the dataset consisting of *28 x 28* images. I flatten these images into one array with *28 × 28 = 784* elements. This means that the input layer will have 784 nodes.
2. **Hidden Layer:** In this layer, I have decided to reduce the number of nodes from 784 in the input layer to 64 nodes.
3. **Output Layer:** In this layer, I reduce the 64 nodes to a total of 10 nodes so that I can evaluate the nodes against the label. This label is received in the form of an array with 10 elements, where one of the elements is 1 while the rest are 0.

### Five Layer Model (Three Hidden Layers)
The input and output layers are same as the Simple Three Layer Model.
- **First Hidden Layer:** In this layer, I have decided to reduce the number of nodes from 784 in the input layer to 128 nodes.
- **Second Hidden Layer:** In this layer, I decided to go with 64 nodes, from the 128 nodes in the first hidden layer. 
- **Third Hidden Layer:** Finally, I decided to go with 32 nodes, from the 64 nodes in the second hidden layer. 

You probably realize that the number of nodes in each layer decreases from 784 input nodes to 128 nodes to 64 nodes to 32 nodes to 10 output nodes. This is based on empirical observations that this yields better results because we are not overfitting nor underfitting, only trying to get just the right number of nodes. The specific number of nodes chosen for this article were chosen at random, although decreasing to avoid overfitting. In most real-life scenarios, you would want to optimize these parameters by brute force or good guesses, usually by grid search or random search, but this is outside the scope of this assigned task.

## Imports and Dataset Preparation
Since I want to build the entire Neural Network using Python and its NumPy library only, I imported train and test data using NumPy *loadtxt* method. It takes relatively more time than Pandas library and load the csv files in numpy ndarray format which must be preprocessed before feeding them to the neural netowrk model.

For this purpose, I created a **CustomDataLoader** class. I created a constructor for this class that takes parameters:
- **data_dir:** directory path to the mnist dataset, *data* in my case.
- **image_size:** size of image in the dataset, *28* in my case.
- **label_num:** number of labels (classes) in the dataset, *10* in my case.
- **seed:** random seed to generate random numbers, *42* in my case.

Beside these parameters, the constructor __init__ method initializes three more attributes. They are:
- **pixel_size:** size of each pixel i.e. *image_size x image_size*
- **train_data:** loaded numpy ndarray of MNIST train data.
- **test_data:** loaded numpy ndarray of MNIST test data.

```python
def __init__(self, data_dir=None, image_size=None, label_num=None, seed=None):
    self.data_dir = data_dir if data_dir is not None else DATA_DIR_PATH
    self.image_size = image_size if image_size is not None else IMAGE_SIZE
    self.label_num = label_num if label_num is not None else LABEL_NUM
    self.seed = seed if seed is not None else SEED
    self.pixel_size = self.image_size * self.image_size
    self.train_data = np.loadtxt(os.path.join(self.data_dir, TRAIN_DATA), delimiter=',', skiprows=1)
    self.test_data = np.loadtxt(os.path.join(self.data_dir, TEST_DATA), delimiter=',', skiprows=1)
```

Then, I created a method to one-hot encode the target labels for classification. Subtracting these one-hot encoded labels from the output of the neural network is more subtle in the training algorithm used in neural network model.

```python
def one_hot_encode(self, y):
    encoded = np.zeros((10, 1))
    encoded[y] = 1.0
    return encoded
```

After that, I created another method to pre-process the loaded datasets and turn them into a trainable and testable data for my training algorithm.

First images and labels were extracted from each train and test data using slicing. Then, train and test images were normalized by dividing all images by 255 and made it such that all images have values between 0 - 1 because this removes some of the numerical stability issues with activation functions later on. Finally, train_images and train_labels are randomly shuffled.

After that, X values of each data are reshaped into list of pixel size of the image, 784 in my case. Likewise, y values are transformed into their one-hot encoded vectors.

Finally the X and y values for each train and test data are zipped separately and returned as a tuple of (training_data, testing_data)

```python
def preprocess_data(self):

    train_images = np.asfarray(self.train_data[:, 1:]) * SCALE_FACTOR
    test_images = np.asfarray(self.test_data[:, 1:]) * SCALE_FACTOR

    train_labels = np.asfarray(self.train_data[:, :1])
    test_labels = np.asfarray(self.test_data[:, :1])

    np.random.seed(self.seed)
    shuffle_index = np.random.permutation(train_images.shape[0])
    train_images, train_labels = train_images[shuffle_index], train_labels[shuffle_index]

    # Transforming X values into a list of pixel_size i.e. 28*28=784 if image_size=28
    X_train = [np.reshape(x, (self.pixel_size, 1)) for x in train_images]
    X_test = [np.reshape(x, (self.pixel_size, 1)) for x in test_images]

    # Transforming y values into their one-hot encoded vector form
    y_train = [self.one_hot_encode(y) for y in train_labels.astype(int)]
    y_test = [self.one_hot_encode(y) for y in test_labels.astype(int)]

    # Zipping the training data and testing data
    training_data = zip(X_train, y_train)
    testing_data = zip(X_test, y_test)

    return (training_data, testing_data)
```

## Setting Up settings.py Module

This module contains all the constant variables that are needed to create and run my neural network model.

1. The first one is directory path to dataset in the project folder. I have used the in-built *os* python library to get the path to the directory of dataset in my current project directory.
2. Then, I declared variable names for each train and test dataset with their extension .csv in string.
3. Then, I declared the size of image used in load_mnist_data.py module.
4. After that, I declared the scale factor to normalize the image pixels of train and test data.
5. Then, I declared the seed value to generate random numbers.
6. Then, I declared hyperparameters for the training algorithm I used for my neural network. They are *mini_batch_size*, *epochs* and *learning_rate*.
7. Finally, I declared the two neural network sizes with different number of layers for the different test cases on my neural network model. 

```python
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
ETA = 0.4  # Learning rate

# List of number of neurons per layer in our neural network
# Input neuron size must be pixel_size = IMAGE_SIZE*IMAGE_SIZE 
# Output neuron size must be the number of labels (target_value)
THREE_LAYER_NN = [784, 30, 10] # Three layers (one input, one hidden, one output)
THREE_HIDDEN_LAYER_NN = [784, 128, 64, 32, 10] 
```

## Setting Up utils.py Module

Here, I have created helper functions for my neural network model. They are explained below:

1. **Activation Function**

   The purpose of an activation function is to add non-linearity to the neural network. As I have mentioned earlier, I have implemented popular Sigmoid activation function and a modified sigmoid function with positive *z* to its exponent part.
   - **Sigmoid:** This function takes any real value as input   and outputs values in the range of 0 to 1. The larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to 0.0

    $$ \sigma (z) =  \frac{\mathrm{1} }{\mathrm{1} + e^\mathrm{-z} } $$

    The derivative of sigmoid function can be written as:
    $$ \sigma' (z) =  { \sigma (z) (\mathrm{1} - \sigma (z)) } $$ 

    This function is used in backpropagation to calculate the *delta* or gradient.

    ```python
    def sigmoid(z, derivative=False):
        if derivative:
            # return (np.exp(-z))/((1.0+np.exp(-z))**2)
            return sigmoid(z)*(1-sigmoid(z))
        return 1.0/(1.0+np.exp(-z))
    ```

   - **Alternative Sigmoid:** This function also takes any real value as input and outputs values in the range of 0 to 1. Since the exponent term is positive, and the graph seems to be reflection of original sigmoid function on Y-axis, the larger the input (more positive), the closer the output value will be to 0.0, whereas the smaller the input (more negative), the closer the output will be to 1.0.

    $$ \sigma (z) =  \frac{\mathrm{1} }{\mathrm{1} + e^z } $$

    Its derivation can be written as:
    $$ \sigma' (z) =  { -\sigma (z) (\mathrm{1} - \sigma (z)) } $$

    ```python
    def altered_sigmoid(z, derivative=False):
        if derivative:
            # return (-np.exp(z))/(1.0+np.exp(z)**2)
            return -altered_sigmoid(z)*(1 - altered_sigmoid(z))
        return 1.0/(1.0+np.exp(z))
    ```

2. **Cost Function**

   Cost function calculates the model error and used to evaluate the model.Mean Squared Error (MSE) and Cross-entropy are the two main types of cost functions to use. Here, I have used Mean Squared Error (MSE) which is represented by the following quadratic function.

    $$ C(w, b) = {\frac{1}{2n} \Sigma_{x}||{y(x)}-a||^2} $$

    Here *w* is a weight, *n* is the total number of input images, *b* is the bias, y(x) is the true label, and a is the prediction of your neural network, or it could be an immediate output of the particular layer.

    Its derivative w.r.t the activation of the output layer (prediction) can be simply obtained as:

    $$ {\frac{\delta C}{\delta a}} = {y(x)}-a $$

   ```python
    def mse_cost_function(y_true, y_pred, derivative=False):
        n = y_pred.shape[1]
        if derivative:
            cost_derivative = y_pred - y_true
            return cost_derivative
        cost = (1.0/(2*n)) * np.sum((y_true - y_pred) ** 2)
        return cost
   ```

## Building Neural Network
I have created a **NeuralNetwork** class upon which all methods for the model architecture of my neural network follows. This class has a constructor that takes parameters:
- **sizes:** a list of the number of neurons (activations) in each layer.
- **activation:** activation function of the neural network, default value is sigmoid.
- **seed:** random seed to generate random numbers. 

Besides, the constructor __init__ method initializes three more instance attributes:
- **num_layers:** length of the sizes
- **weights:** the weights connecting each node, randomized for each connection between the input and output layers. 
- **biases:** the initial biases of our network, randomized for each layer after the input layer. 

The initialization of weights in the neural network is actually quite difficult to think about. To really understand how and why the following approach works, you need a grasp of linear algebra, specifically dimensionality when using the dot product operation.

The specific problem that arises when trying to implement the feedforward neural network is that we are trying to transform from 784 nodes to 10 nodes.

Let us analyze at how the *sizes* affect the parameters of the neural network when initializing the weights and biases. I am preparing *m x n* matrices that are "dot-able" so that I can do a forward pass, while shrinking the number of activations as the layers increase. I can only use the dot product operation for two matrices M1 and M2, where *m* in M1 is equal to n in M2, or where *n* in M1 is equal to *m* in M2.

```python
def __init__(self, sizes, activation='alternative', cost_function='mse', seed=42):

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

    # Instantiate all weights and biases randomly
    self.weights = [np.random.randn(y, x)*np.sqrt(1/x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
    self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
```

## Feedforward Function
The forward pass consists of the dot operation in NumPy, which turns out to be just matrix multiplication. It can be represented as:
    $$ y = {\sigma (wX + b)} $$

The feedforward function is the function that sends information forward in the neural network. This function will take one parameter, *a*, representing the current activation vector. This function loops through all the biases and weights in the network and calculates the activations at each layer. The *a* returned is the activations of the last layer, which is the prediction.

```python
def feed_forward(self, a):
    for w, b in zip(self.weights, self.biases):
        z = np.dot(w, a) + b
        a = self.activation(z)
    return a
```

## Backpropagation
Backpropagation is the updating of all the weights and biases after we run a training epoch. We use all the mistakes the network makes to update the weights. 

The backward pass is hard to get right because there are so many sizes and operations that must align for all of the operations to be successful. This method use the derivative of the cost function, MSE in my case and also the derivative of the activation function to update weights and biases with backward pass.

My backpropagation function has two parts in it:
1. **Feeding Forward:** 
   
   It will take two values: x, and y. Firstly, I have initialized the gradients($\nabla$) of weights and biases. This symbol represents the gradients. We also need to keep track of our current activation vector, *activation*, all of the activation vectors, *activations*, and the *z*-vectors or pre-activation vectors, *pre-activations*. The first activation is the input layer.
   
   After setting these up, I looped through all the biases and weights. In each loop, I have calculated the *z* vector as the dot product of the weights and activation, added that to the list of *pre-activations*, recalculated the activation, and then add the new activation to the list of *activations*.

2. **Backward Pass:** 
   
   Now comes the role of calculus. I started my backward pass by calculating the *delta*, which is equal to the error from the last layer multiplied by the derivative of sigmoid function of the last entry of the *pre-activations* vectors. 
   
   I set the last layer of *del_bias* as the *delta* and the last layer of *del_weight* equal to the dot product of the *delta* and the second to last layer of *activations* (transposed so I can actually do the math). After setting these last layers up, I did the same thing for each layer going backwards starting from the second to last layer. Finally, the updated *del_weight* and *del_bias* are returned as a tuple.

```python
def back_propagate(self, x, y):

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
```

## Mini-Batch Update
It starts much the same way as my *back_propagate* method by creating 0 vectors of the gradients for the weights and biases (*del_weight*, *del_bias*). It takes four parameters: the *mini_batch*, the *learning* rate, an empty list of *train_losses*, and an empty list of *train_accuracies*.

Then, for each input, *x*, and output, *y*, in the *mini_batch*, I obtained the *delta* of each gradients array via the *back_propagate* function. Next, I updated the gradient lists with these *deltas*. Finally, I updated the weights and biases of the network using the updated gradients list and the learning rate. Each value is updated to the current value minus the learning rate divided by the size of the mini_batch times the updated gradient value.

To obtain the value of cost function (MSE), loss in my case, I first obtained predicted value using *predict* method each loop in mini_batch and then computed the cost function using this predicted value (output of activation function) and ground truth value (y in *mini_batch*). Finally, I appended this value to the empty list *train_losses* passs as the argument in this *update_mini_batch* function.

I followed same process to obtain the training accuracy for each predicted value using accuracy_score method and appending to the empty list *train_accuracies* passed as the argument in this *update_mini_batch* function. 

### Code for update_mini_batch function
```python
def update_mini_batch(self, mini_batch, learning_rate, train_losses, train_accuracies):

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
```

### Code for predict helper function

```python
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
```

### Code for accuracy_score helper function

```python
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
``` 

## Training with Mini-Batch Gradient Descent Algorithm

I have defined a forward and backward pass, but how can I start using them? I must make a training loop and an optimizer to update the parameters of the neural network. The base optimizer for training any machine learning model is Gradient Descent. Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function.

I have also applied this algorithm but with some alterations. In particular, I have implemented Mini-Batch Gradient Descent, a modified version of traditional Gradient Descent algorithm.

Mini-Batch Gradient Descent is actually a mixture of Batch Gradient Descent and Stochastic Gradient Descent algorithms. These are again other two popular modified versions of Gradient Descent.

In Batch Gradient Descent, all the training data is taken into consideration to take a single step. We take the average of the gradients of all the training examples and then use that mean gradient to update our parameters. Contrary to Batch Gradient Descent in Stochastic Gradient Descent (SGD), we consider just one example at a time to take a single step. Batch Gradient Descent converges directly to minima. SGD converges faster for larger datasets. But, since in SGD we use only one example at a time, we cannot implement the vectorized implementation on it. This can slow down the computations. To tackle this problem, Mini-Batch Gradient Descent is often preferred.

In Mini-Batch Gradient Descent, neither we use all the dataset all at once nor we use the single example at a time. We use a batch of a fixed number of training examples which is less than the actual dataset and call it a mini-batch. Doing this helps us achieve the advantages of both the former variants - Batch Gradient Descent and Stochastic Gradient Descent.

### Algorithm

To implement the Mini-Batch Gradient Descent algorithm, I have created *mini_batch_gradient_descent* method that takes four mandatory parameters and one optional parameter. The four mandatory parameters are:
- **train_data:** input data that is used for training
- **mini_batch_size:** number of data points to process in each batch
- **epochs:** number of epochs (number of times we run through the whole dataset) for the training
- **learning_rate:** value of the learning rate of the model

The optional parameter is for test_data. It can be passed optinally into the algorithm if we want to test the neural network with the test data created in *load_mnist_data* module.

In this method, I first created empty lists for storing histories of training losses, training accuracies and test accuracies. Then, I converted the *train_data* into a list type and set the number of samples to the length of that list. If the test data is passed in, same processes are followed. This is because these are not returned as lists, but zips of lists. Note that this type-casting isn’t strictly necessary if we can ensure that we pass both types of data in as lists.

There are two main loops in my training algorithm. One loop for the number of training epochs, which is the number of times I ran through the entire data set. In each epoch, I started by shuffling the data to ensure randomness, then I created a list of mini-batches. The second loop for running through each mini-batch by calling the update_mini_batch method defined earlier.

In update_mini_batch method, the *train_losses* and *train_accuracies* are updated with training losses and training acccuracies for each data in mini-batch. For each epoch, the mean of *train_losses* list values and *train_accuracies* list values are displayed and appended to the *history_train_losses* and *history_train_accuracies* respectively. 

Besides, if test_data is passed into the training algorithm, the neural network model is evaluated with the test data by calling the *evaluate* method to get the test acccuracy. This part is discussed in the next section - **Evaluation**. Then, the test_accuracy is appended to history_test_accuracies list and displayed with training loss and training accuracy for each epoch.

```python
def mini_batch_gradient_descent(self, train_data, mini_batch_size, epochs, learning_rate, test_data=None):

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
```

## Evaluation
The last class method I created is *evaluate*, that evaluates my neural network model on test data on each epoch by calculating the accuracy i.e. number of correct prediction over all test data. 

This method takes one parameter, the test_data. In this method, I have simply compared the network’s outputs (from *feed_fowrward* function) with the ground truth value, y. The network’s outputs are calculated by feeding forward the input, x.

```python
def evaluate(self, test_data):
    test_results = [(np.argmax(self.feed_forward(x)), y)
                    for (x, y) in test_data]
    accuracy = (sum(int(y[x]) for (x, y) in test_results) / len(test_data))
    return accuracy
```

## Testing My Neural Network
To run my training algorithm and test my neural network model on test data, I have created a file named main.py. It will import all the modules created so far except *utils* i.e. *neural_net*, *load_mnist_data* and *settings*. 

Firstly, I created an instance of my **CustomDataLoader** class and called preprocess_data method to get the tuple of training and testing data for my training algorithm.

Then, I created an instance of my **NeuralNetwork** class with *sizes* equals to THREE_LAYER_NN in *settings* module. The reason behind this is I wanted to test data on my three layer neural network. After that, to initialize the training and testing I called the training method *mini_batch_gradient_descent* with *epochs*, *batch_size* and *learning_rate* defined in *settings* module.

I repeated the same process to test my neural network model with *sizes* equals to THREE_HIDDEN_LAYER_NN in *settings* module. 

Note that it doesn’t matter what any of the values in between input size = *28 x 28* = 784 and output size = 10 (number of target labels) are for our list of input layers. Only the input size and output size are set, we can adjust the rest of the hidden layers size however we like.

## Outputs
With original sigmoid activation function, the training accuracy reached to $~0.97$ and the test accuracy reached to $~0.93$, while training loss reached to $~0.13$ with epoch = 10, batch_size = 10, learning_rate $(\eta)$ = 0.04 and neural net size THREE_LAYER_NN = $[728. 64, 10]$. Not bad right?
```bash
Epoch 0 / 10 | train_loss: 0.385 | train_accuracy: 0.923 | test_accuracy : 0.844 
Epoch 1 / 10 | train_loss: 0.2769 | train_accuracy: 0.9446 | test_accuracy : 0.89 
Epoch 2 / 10 | train_loss: 0.2261 | train_accuracy: 0.9548 | test_accuracy : 0.9008 
Epoch 3 / 10 | train_loss: 0.1969 | train_accuracy: 0.9606 | test_accuracy : 0.9077 
Epoch 4 / 10 | train_loss: 0.1777 | train_accuracy: 0.9645 | test_accuracy : 0.913 
Epoch 5 / 10 | train_loss: 0.1638 | train_accuracy: 0.9672 | test_accuracy : 0.9179 
Epoch 6 / 10 | train_loss: 0.1532 | train_accuracy: 0.9694 | test_accuracy : 0.921 
Epoch 7 / 10 | train_loss: 0.1448 | train_accuracy: 0.971 | test_accuracy : 0.9231 
Epoch 8 / 10 | train_loss: 0.1379 | train_accuracy: 0.9724 | test_accuracy : 0.9257 
Epoch 9 / 10 | train_loss: 0.132 | train_accuracy: 0.9736 | test_accuracy : 0.9272 
```

### Experimenting With the Altered Sigmoid Function
When the altered sigmoid function was used for training, it surprisingly worked and gives result for loss and accuracy similar to its original version. I got surprised with this result, because back then I expected it would give increasing loss values and decreasing accuracy values in each iteration on epoch while training the model and testing on the test_data. 
```bash
Epoch 0 / 10 | train_loss: 0.3812 | train_accuracy: 0.9238 | test_accuracy : 0.8463 
Epoch 1 / 10 | train_loss: 0.2744 | train_accuracy: 0.9451 | test_accuracy : 0.8895 
Epoch 2 / 10 | train_loss: 0.2244 | train_accuracy: 0.9551 | test_accuracy : 0.9012 
Epoch 3 / 10 | train_loss: 0.1957 | train_accuracy: 0.9609 | test_accuracy : 0.9087 
Epoch 4 / 10 | train_loss: 0.1768 | train_accuracy: 0.9646 | test_accuracy : 0.9129 
Epoch 5 / 10 | train_loss: 0.1632 | train_accuracy: 0.9674 | test_accuracy : 0.9173 
Epoch 6 / 10 | train_loss: 0.1528 | train_accuracy: 0.9694 | test_accuracy : 0.9204 
Epoch 7 / 10 | train_loss: 0.1445 | train_accuracy: 0.9711 | test_accuracy : 0.923 
Epoch 8 / 10 | train_loss: 0.1377 | train_accuracy: 0.9725 | test_accuracy : 0.9241 
Epoch 9 / 10 | train_loss: 0.1319 | train_accuracy: 0.9736 | test_accuracy : 0.9268 
```

It may happened because this function is just a reflection of original sigmoid function on Y-axis. Here, unlike the original sigmoid function, the larger the input (more positive), the closer the output value will be to 0.0, whereas the smaller the input (more negative), the closer the output will be to 1.0.

The derivative of this function can also be represented same as original sigmoid function but with negative sign, which means it has same nature as derivative of sigmoid function but has inverted graph i.e. inverted Gaussian Distribution. This means, the label classified by sigmoid might be represented by inverted label by its altered version i.e. label 0 in original sigmoid = label 1 in altered sigmoid. Also the classification is not dependent on the domain values i.e. values of x. It can range from $[-\infty, \infty]$, but the both original sigmoid and its altered version with positive $z$ gives the output with range $[0, 1]$. Therefore, classification with probability prediction value with boundary plane on 0.5 is possible for both activation functions.

### Adding Two More Hidden Layers
The training time increased with all hyperparameters same as for simple three layer neural network, which is not doubtful since the number of hidden layers have increased which means more computations need to be done while backpropagating the gradient values of weights and biases on each iteration on mini_batch of training data. 

The second signification observation is on loss values which tends to be large for all epoch iterations. Thus, although the training accuracy could reach the value of $~0.91$ the model showed poor test accuracy score with last value reaching $~0.71$. 
```bash
Epoch 0 / 10 | train_loss: 0.5033 | train_accuracy: 0.8993 | test_accuracy : 0.1135 
Epoch 1 / 10 | train_loss: 0.5016 | train_accuracy: 0.8997 | test_accuracy : 0.1138 
Epoch 2 / 10 | train_loss: 0.5011 | train_accuracy: 0.8998 | test_accuracy : 0.1135 
Epoch 3 / 10 | train_loss: 0.5008 | train_accuracy: 0.8998 | test_accuracy : 0.1883 
Epoch 4 / 10 | train_loss: 0.5007 | train_accuracy: 0.8999 | test_accuracy : 0.1535 
Epoch 5 / 10 | train_loss: 0.5005 | train_accuracy: 0.8999 | test_accuracy : 0.2637 
Epoch 6 / 10 | train_loss: 0.5002 | train_accuracy: 0.9 | test_accuracy : 0.3261 
Epoch 7 / 10 | train_loss: 0.4926 | train_accuracy: 0.9015 | test_accuracy : 0.4638 
Epoch 8 / 10 | train_loss: 0.4803 | train_accuracy: 0.9039 | test_accuracy : 0.5943 
Epoch 9 / 10 | train_loss: 0.4625 | train_accuracy: 0.9075 | test_accuracy : 0.712 
```
The reason behind this might be the use of sigmoid activation function for the ouput prediction. In my case, I am doing classification of handwritten digits (0-9) which is a multi-class classification task. Sigmoid activation is more effective on binary classification task since it tries to classify based on the boundary plane on 0.5 value and classifies labels as either 1 or 0 i.e. binary type. Although, I used one-hot encoding to achive miulti-class classification, it is not enough for better performance of the model.

I could apply other popular activation functions such as *Softmax* which is highly effective for multi-class classification task and thus for the output prediction in our case, *ReLU* or *Leaky ReLU* which are highly effective for activation function in input and hidden layers. 

Besides, I could use other effective optimizer like Adam, RMSProp, etc. for better optimization. Also, the cost function MSE can be substituted with Cross Entropy which might be effective on my case.

Apart from these, there is always room to play with hyperparameters (epochs, batch_size, learning_rate, number of neurons on each layers) and tune them as we like to get the best result and performance from the model.

## References
1. [Training and Testing with MNIST](https://python-course.eu/machine-learning/training-and-testing-with-mnist.php)
2. [CS565600 Deep Learning](https://nthu-datalab.github.io/ml/labs/10_TensorFlow101/10_NN-from-Scratch.html), National Tsing Hua University
3. [Building a Neural Network from Scratch: Part 1](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/)
4. [Building a Neural Network from Scratch: Part 2](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%202/)
5. [MNIST Handwritten digits classification from scratch using Python Numpy.](https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab)