import matplotlib.pyplot as plt     # Plotting
import numpy as np                  # Matrix Math

class Network():
   
    def __init__(self, layer_sizes=[1, 64, 64, 64, 1], activation_function='tanh'):

        # Specify NN architecture architecture:
        self.layer_sizes = layer_sizes # [input size, hidden 1, ..., output size]
        self.num_layers = len(layer_sizes) - 1
        self.activation_fun = activation_function # options are relu and tanh

        # Initalize all weights and biases of the network with small random values:
        self.params = dict()
        self.param_init_scale = 0.1
        for layer in range(1,len(self.layer_sizes)):
            self.params[f'W{layer}'] = self.param_init_scale * np.random.rand(self.layer_sizes[layer-1], self.layer_sizes[layer]) - self.param_init_scale/2
            self.params[f'b{layer}'] = self.param_init_scale * np.random.rand(1, self.layer_sizes[layer]) - self.param_init_scale/2
        
        
    def forwardprop(self, input, target):
    # Forward propagation & Loss function

        cache = dict()
        m = input.shape[0]

        # Input layer:
        a = input
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

            a *= self.dropout[f'a{layer}'] / (1-self.dropout['drop_prob']) # dropout mask

            cache[f"z{layer}"] = z
            cache[f"a{layer}"] = a

        # Output layer:
        y = np.dot(a, self.params[f'W{self.num_layers}']) + self.params[f'b{self.num_layers}']

        # Cost function:
        L = np.power(y - target, 2)

        cache['y'] = y
        cache['L'] = L
        cache['target'] = target

        return L, cache

    
    def backprop(self, cache):
    # Backward pass algorithm for computing parameter gradients w.r.t. Loss function
        
        grads = dict()
        m = cache['y'].shape[0]

        # Cost function gradient:
        dz = 2*(cache['y'] - cache['target'].reshape(-1,1))

        # Output layer gradient:
        grads[f'dW{self.num_layers}'] = (1/m)*np.dot(cache[f'a{self.num_layers-1}'].T, dz)
        grads[f'db{self.num_layers}'] = (1/m)*np.sum(dz, axis=0).reshape(1,-1)
        
        # Hidden layer gradients:
        for layer in reversed(range(1,self.num_layers)):
            da = np.dot(dz, self.params[f'W{layer+1}'].T)
            da *= self.dropout[f'a{layer}']/(1-self.dropout['drop_prob']) # dropout mask
            if self.activation_fun == 'tanh':
                dz = da * (1 - np.power(np.tanh(cache[f'z{layer}']),2))
            elif self.activation_fun == 'relu':
                dz = da * (cache[f'z{layer}'] > 0)
            grads[f'dW{layer}'] = (1/m)*np.dot(cache[f'a{layer-1}'].T, dz)
            grads[f'db{layer}'] = (1/m)*np.sum(dz, axis=0).reshape(1,-1)

        return grads

    def refresh_dropout(self, drop_prob=0):

        self.dropout = dict()
        for layer in range(1,self.num_layers):
            self.dropout[f'a{layer}'] = (drop_prob < np.random.rand(1, self.layer_sizes[layer]))

        self.dropout['drop_prob'] = drop_prob


    def visualize_network(self):

        fig = plt.figure()
        fig.set_size_inches(10,4)
        for i in range(1,self.num_layers+1):
            plt.subplot(1,self.num_layers,i)
            plt.plot(self.params[f'W{i}'].reshape((-1,)))
            plt.title(f"W{i}")
        plt.show()

        fig = plt.figure()
        fig.set_size_inches(10,4)
        for i in range(1,self.num_layers+1):
            plt.subplot(1,self.num_layers,i)
            b = self.params[f'b{i}'].reshape((-1,))
            if i == self.num_layers:
                b = [b, b]
            plt.plot(b)
            plt.title(f"b{i}")
        plt.show()