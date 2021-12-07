import matplotlib.pyplot as plt     # Plotting
import numpy as np                  # Matrix Math

class Network():

    # Specify NN architecture architecture:
    layer_sizes = [] # [input size, hidden 1, ..., output size]
    num_layers = 0
    activation_fun = '' # options are relu and tanh

    param_init_scale = 0.01

    params = dict()
    grads = dict()

    
    def __init__(self, layer_sizes=[1, 64, 64, 64, 1], activation_function='tanh'):

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        self.activation_fun = activation_function

        # Initalize all weights and biases of the network with small random values:
        for layer in range(1,len(self.layer_sizes)):
            self.params[f'W{layer}'] = self.param_init_scale * np.random.rand(self.layer_sizes[layer-1], self.layer_sizes[layer])
            self.params[f'b{layer}'] = self.param_init_scale * np.random.rand(1, self.layer_sizes[layer])
        
    def forwardprop(self, input, target):
    # Forward propagation & Loss function

        cache = dict()

        # Input layer:
        a = input.reshape(-1,1)
        cache['a0'] = a

        # Hidden layers:
        for layer in range(1,self.num_layers):
            W = self.params[f'W{layer}']
            b = self.params[f'b{layer}']

            z = np.dot(a, W) + b
            if self.activation_fun == 'tanh':
                a = np.tanh(z)
            elif self.activation_fun == 'relu':
                a = np.maximum(z,0)

            cache[f"z{layer}"] = z
            cache[f"a{layer}"] = a

        # Output layer:
        y = np.dot(a, self.params[f'W{self.num_layers}']) + self.params[f'b{self.num_layers}']

        # Cost function:
        L = np.power((y - target.reshape(-1,1)),2)

        cache['y'] = y
        cache['L'] = L
        cache['target'] = target

        return L, cache

    
    def backprop(self, cache):
    # Backward pass algorithm for computing parameter gradients w.r.t. Loss function
        
        grads = dict()

        # Cost function gradient:
        dz = 2*(cache['y'] - cache['target'].reshape(-1,1))

        # Output layer gradient:
        grads[f'dW{self.num_layers}'] = np.dot(cache[f'a{self.num_layers-1}'].T, dz)
        grads[f'db{self.num_layers}'] = dz
        
        # Hidden layer gradients:
        for layer in reversed(range(1,self.num_layers)):
            da = np.dot(dz, self.params[f'W{layer+1}'].T)
            if self.activation_fun == 'tanh':
                dz = da * (1 - np.power(np.tanh(cache[f'z{layer}']),2))
            elif self.activation_fun == 'relu':
                dz = da * (cache[f'z{layer}'] > 0)
            grads[f'dW{layer}'] = np.dot(cache[f'a{layer-1}'].T, dz)
            grads[f'db{layer}'] = dz

        return grads


    def visualize_network(self):

        fig = plt.figure()
        fig.set_size_inches(10,4)
        for i in range(1,self.num_layers):
            plt.subplot(1,4,i)
            plt.plot(self.params[f'W{i}'].reshape((-1,)))
            plt.title(f"W{i}")
        plt.show()

        fig = plt.figure()
        fig.set_size_inches(10,4)
        for i in range(1,self.num_layers):
            plt.subplot(1,4,i)
            plt.plot(self.params[f'b{i}'].reshape((-1,)))
            plt.title(f"b{i}")
        plt.show()