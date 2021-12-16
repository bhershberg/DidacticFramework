from re import I
import matplotlib.pyplot as plt     # Plotting
import numpy as np
from numpy.lib.function_base import kaiser                  # Matrix Math
from tqdm import tqdm               # Progress Bar
from IPython import display

class Optimizer():

    def __init__(self, optimizer_method="ADAM", learning_rate=1e-3, beta1=0.9, beta2=0.999, num_epochs=100, batch_size=64, drop_prob=0, drop_decay_rate=0):

        self.optimizer_method = optimizer_method # Supported options are SGD and ADAM
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size=batch_size

        # Adam hyperparameters
        self.epsilon = 1e-8
        self.m = dict()
        self.v = dict()

        # Dropout hyperparameters
        self.drop_prob = drop_prob
        self.drop_decay_rate = drop_decay_rate

        self.loss_history = []

    def train_network(self, Network, Loader, grad_checking=False, visualize=True):
    # Main loop(s) used for training the network over all epochs and samples

        num_iters = int(np.ceil(Loader.x_train.shape[0] / self.batch_size))
        self.Loss = 0
        drop_decay = self.drop_decay_rate*self.drop_prob/self.num_epochs
        self.beta1pow = self.beta1
        self.beta2pow = self.beta2

        print("Training:")

        t = tqdm(range(self.num_epochs))
        for epoch in t:

            # setup
            num_grad_checks = 1
            if self.drop_decay_rate > 0:
                self.drop_prob = max(self.drop_prob - drop_decay, 0)

            # main training loop
            for _ in range(num_iters):
                Network.refresh_dropout(drop_prob=self.drop_prob)
                (x_batch, y_batch) = Loader.get_train_batch(batch_size=self.batch_size)
                self.Loss, cache = Network.forwardprop(x_batch, y_batch)
                self.loss_history.append(np.mean(np.array(self.Loss)))
                grads = Network.backprop(cache)
                if grad_checking and num_grad_checks > 0: # optional gradient checking
                    num_grad_checks -= 1
                    self.gradient_checking(Network, grads, x_batch, y_batch)
                self.update_params(Network, grads)

            t.set_description(f"Loss={np.mean(np.array(self.loss_history[-num_iters:]))}")

        return Network.params, self.Loss

    def update_params(self, Network, grads):
    # Learning rules are applied here to update actual network parameters (weights/biases)

        if self.optimizer_method.upper() == 'SGD':

            for param_name in Network.params:
                Network.params[param_name] -= self.learning_rate * grads[f'd{param_name}']

        elif self.optimizer_method.upper() == 'ADAM':

            for param_name in Network.params:

                if param_name not in self.m: # If not yet initialized, do so
                    self.m[param_name] = np.zeros_like(Network.params[param_name])
                    self.v[param_name] = np.zeros_like(Network.params[param_name])

                self.m[param_name] = self.beta1 * self.m[param_name] + (1-self.beta1) * grads[f"d{param_name}"]
                self.v[param_name] = self.beta2 * self.v[param_name] + (1-self.beta2) * np.square(grads[f"d{param_name}"])
                self.m_hat = self.m[param_name] / (1-self.beta1)
                Network.params[param_name] -= self.learning_rate * self.m[param_name] / (np.sqrt(self.v[param_name]) + self.epsilon)

            #     # The full Adam algorithm that adjusts the initial exponential moving average 
            #     # doesn't make too much difference in training outcome, but slows it down...
            #     m_hat = self.m[param_name] / (1-self.beta1pow)
            #     v_hat = self.v[param_name] / (1-self.beta2pow)
            #     Network.params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            # self.beta1pow *= self.beta1pow
            # self.beta2pow *= self.beta2pow
        
        return Network.params

    def gradient_checking(self, Network, grads, inputs, targets):

        # This is a non-exhaustive gradient checker. 
        # It only checks one gradient per parameter array and assumes that if that's correct, they're all correct.
        for param_name in Network.params:
            epsilon = 1e-8
            i, j = [0, 0]
            tmp = Network.params[param_name][i][j]
            Network.params[param_name][i][j] = tmp + epsilon
            Lplus, _ = Network.forwardprop(inputs, targets)
            Network.params[param_name][i][j] = tmp - epsilon
            Lminus, _ = Network.forwardprop(inputs, targets)
            Network.params[param_name][i][j] = tmp # restore originial value
            dLest = (Lplus - Lminus) / (2*epsilon)

            dParam = grads[f"d{param_name}"]
            passFail = 'PASS'
            if(not np.isclose(dParam[i][j],np.mean(dLest), rtol=1e-1)):
                passFail = 'FAIL'
                print(f"{param_name}: grad={dParam[i][j]}, grad_check={np.mean(dLest)}  {passFail}")


    def test_network(self, Network, Loader):

        Network.refresh_dropout(drop_prob=0)
        self.predictions = np.array(()).reshape(-1,1)
        (x_test, y_test) = Loader.get_test_batch(batch_size=-1) # batch_size=-1 grabs the entire test set
        L, cache = Network.forwardprop(x_test, y_test)
        self.predictions = cache['y'].reshape(-1,1)

        fig = plt.figure()
        fig.set_size_inches(10,4)
        plt.subplot(1,2,1)
        plt.scatter(x_test, y_test, label='Target')
        plt.scatter(x_test, self.predictions, label='Prediction')
        plt.title("Function Fit")
        plt.legend()

        plt.subplot(1,2,2)
        plt.semilogy(self.loss_history)
        plt.xlabel("Mini Batch #")
        plt.ylabel("Loss")
        plt.title("Training Set Loss")

        plt.show()

        print(f"Test Set Loss: {np.mean(L)}")

        return self.predictions