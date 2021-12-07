import matplotlib.pyplot as plt     # Plotting
import numpy as np                  # Matrix Math
from tqdm import tqdm               # Progress Bar

class Optimizer():

    # Adam hyperparameters
    epsilon = 1e-8
    t = 0
    m = dict()
    v = dict()

    loss_history = []

    def __init__(self, optimizer_method="ADAM", learning_rate=1e-3, beta1=0.9, beta2=0.999, num_epochs=100, batch_size=64):

        self.optimizer_method = optimizer_method # Supported options are SGD and ADAM
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size=batch_size


    def train_network(self, Network, inputs, targets, grad_checking=False):
    # Main loop(s) used for training the network over all epochs and samples

        self.Loss = []

        print("Training:")
        for epoch in tqdm(range(self.num_epochs)):
            Lavg = []
            num_checks = 1
            for (input, target) in zip(inputs, targets):
                L, cache = Network.forwardprop(input, target)
                Lavg.append(L)
                grads = Network.backprop(cache)
                if grad_checking and num_checks > 0:
                    num_checks -= 1
                    self.gradient_checking(Network, grads, input, target)
                self.update_params(Network, grads)

            self.loss_history.append(np.mean(np.array(Lavg)))

        return Network.params, self.Loss


    def update_params(self, Network, grads):
    # Learning rules are applied here to update actual network parameters (weights/biases)

        if self.optimizer_method.upper() == 'SGD':

            for param_name in Network.params:
                Network.params[param_name] -= self.learning_rate * grads[f'd{param_name}']

        elif self.optimizer_method.upper() == 'ADAM':

            self.t += 1
            for param_name in Network.params:

                if param_name not in self.m: # If not yet initialized, do so
                    self.m[param_name] = np.zeros_like(Network.params[param_name])
                    self.v[param_name] = np.zeros_like(Network.params[param_name])

                self.m[param_name] = self.beta1 * self.m[param_name] + (1-self.beta1) * grads[f"d{param_name}"]
                self.v[param_name] = self.beta2 * self.v[param_name] + (1-self.beta2) * np.square(grads[f"d{param_name}"])
                self.m_hat = self.m[param_name] / (1-self.beta1)
                Network.params[param_name] -= self.learning_rate * self.m[param_name] / (np.sqrt(self.v[param_name]) + self.epsilon)

                # -- The full version of Adam also corrects for initial conditions. 
                # -- But its not really necessary to include this:
                # m_hat = self.m[param_name] / (1-np.power(self.beta1,self.t))
                # v_hat = self.v[param_name] / (1-np.power(self.beta2,self.t))
                # Network.params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return Network.params

    def gradient_checking(self, Network, grads, input, target):

        for param_name in Network.params:
            epsilon = 1e-8
            i, j = [0, 0]
            tmp = Network.params[param_name][i][j]
            Network.params[param_name][i][j] = tmp + epsilon
            Lplus, _ = Network.forwardprop(input, target)
            Network.params[param_name][i][j] = tmp - epsilon
            Lminus, _ = Network.forwardprop(input, target)
            Network.params[param_name][i][j] = tmp # restore originial value
            dLest = (Lplus - Lminus) / (2*epsilon)

            dParam = grads[f"d{param_name}"]
            passFail = 'PASS'
            if(not np.isclose(dParam[i][j],dLest, rtol=1e-1)):
                passFail = 'FAIL'
                print(f"{param_name}: grad={dParam[i][j]}, grad_check={dLest}  {passFail}")


    def test_network(self, Network, inputs, targets):
        Lavg = []
        self.predictions = []
        for (input, target) in zip(inputs, targets):
            L, cache = Network.forwardprop(input, target)
            Lavg.append(L)
            self.predictions.append(cache['y'].reshape(-1,))

        Loss_test = np.mean(np.array(Lavg))

        fig = plt.figure()
        fig.set_size_inches(10,4)
        plt.subplot(1,2,1)
        plt.scatter(inputs, targets, label='Target')
        plt.scatter(inputs, self.predictions, label='Prediction')
        plt.title("Function Fit")
        plt.legend()

        plt.subplot(1,2,2)
        plt.semilogy(self.loss_history)
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Training Set Loss")
        plt.show()

        print(f"Test Set Loss: {Loss_test}")

        return self.predictions