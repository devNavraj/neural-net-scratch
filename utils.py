import numpy as np


# class Activation:
#     '''
#     Activation function transforms the summed weighted input 
#     from the node into an output value to be fed to the next 
#     hidden layer or as output. 
#     '''
def sigmoid(z, derivative=False):
    '''
    Sigmoid activation function for an array of neuron outputs.
    It handles two modes: normal and its derivative mode.
    Normal mode:        f(z) = 1 / (1 + e^-z)
    Derivative mode:    f'(z) = f(z)(1 - f(z))
    Params:
    ---
    z: Array/single element of neuron outputs before activation
    at layer 1
    ---
    Returns: 
    ---
    An array of the same length as z with the sigmoid activation applied. 
    Point-wise activation on each element of the input z
    '''
    if derivative:
        # return (np.exp(-z))/((1.0+np.exp(-z))**2)
        return sigmoid(z)*(1-sigmoid(z))
    return 1.0/(1.0+np.exp(-z))

def altered_sigmoid(z, derivative=False):
    '''
    Not a sigmoid activation function.
    Similar to sigmoid function but with positive z to e i.e.
    It handles two modes: normal and its derivative mode.
    Normal mode:        f(z) = 1 / (1 + e^z)
    Derivative mode:    f'(z) = -f(z)(1 - f(z))
    ---
    Params:
    ---
    z: Array/single element of neuron outputs before activation
    at layer 1
    ---
    Returns: 
    ---
    An array of the same length as z with the sigmoid activation applied. 
    Point-wise activation on each element of the input z
    '''
    if derivative:
        # return (-np.exp(z))/(1.0+np.exp(z)**2)
        return -altered_sigmoid(z)*(1 - altered_sigmoid(z))
    return 1.0/(1.0+np.exp(z))


# class CostFunction:
#     '''
#     It calculates the model error and used to evaluate the model.
#     Mean Squared Error (MSE) and Cross-entropy are the two main types 
#     of cost functions to use.
#     Here, I have used Mean Squared Error (MSE)
#     '''
def mse_cost_function(y_true, y_pred, derivative=False):
    '''
    Computes the MSE between a ground truth vector 
    and a prediction vector(output of the activation function)
    It handles two modes: normal and its derivative mode.
    Params:
    ---
    y_true: true label vector (real target)
    y_pred: prediction vector (same shape as y_true)

    Returns:
    ---
    mode 1 (No derivative, default)
    cost: a scalar value representing the loss
    ---
    mode 2 (Derivative)
    cost_derivative: derivative of the loss function 
                        w.r.t. the activation of the output
    '''
    n = y_pred.shape[1]
    if derivative:
        cost_derivative = y_pred - y_true
        return cost_derivative
    cost = (1.0/(2*n)) * np.sum((y_true - y_pred) ** 2)
    return cost


    # def cost_derivative(y_true, y_pred):
    #     '''
    #     Computes the derivative of the loss function w.r.t the activation of the output layer
    #     Params:
    #     ---
    #     y_true: true label vector (real target)
    #     y_pred: prediction vector (same shape as y_true)

    #     Returns:
    #     ---
    #     cost_prime: derivative of the loss w.r.t. the activation of the output
    #     '''
    #     cost_derivative = y_pred - y_true
    #     return cost_derivative