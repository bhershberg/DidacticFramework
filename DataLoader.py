import matplotlib.pyplot as plt     # Plotting
import numpy as np                  # Matrix Math

class DataLoader():

    n_x_train = 10000   # the number of training datapoints
    n_x_test = 2000     # the number of testing datapoints
    add_noise = False

    train_index = 0
    test_index = 0

    def generate_dataset(self, func='Basic Sine'):
        # Generate input and output data for the function we want to learn

        if func == 'Basic Sine':

            x_train = np.random.rand(self.n_x_train,1)*18 - 9  # Initialize a vector of with dimensions [n_x, 1] and extend
            y_train = (np.sin(x_train))/2.5           # Calculate the sin of all data points in the x vector and reduce amplitude
            if(self.add_noise): 
                y_train += (np.random.randn(self.n_x_train, 1)/20)  # add noise to each datapoint

            x_test = np.random.rand(self.n_x_test, 1)*18 - 9   # Repeat data generation for test set
            y_test = (np.sin(x_test))/2.5
            if(self.add_noise): 
                y_test += (np.random.randn(self.n_x_test, 1)/20)

        elif func == 'Discontinuity':
            
            x_train = np.linspace(0, 10, self.n_x_train)
            y_train1 = np.power(x_train[:self.n_x_train//2], 1.6)
            y_train2 = np.power(x_train[self.n_x_train//2:], 1.3)
            y_train = np.append(y_train1, y_train2)
            y_train_max = np.max(y_train)
            y_train = y_train / y_train_max

            x_test = np.linspace(0, 10, self.n_x_test)
            y_test1 = np.power(x_test[:self.n_x_test//2], 1.6)
            y_test2 = np.power(x_test[self.n_x_test//2:], 1.3)
            y_test = np.append(y_test1, y_test2)
            y_test = y_test / y_train_max # not a bug, we want to normalize w.r.t. to the same value as the training set!

        # randomize training data:
        random_order = np.random.permutation(np.arange(x_train.size))
        self.x_train = x_train[random_order]
        self.y_train = y_train[random_order]

        random_order = np.random.permutation(np.arange(x_test.size))
        self.x_test = x_test[random_order]
        self.y_test = y_test[random_order]

        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def get_train_batch(self, batch_size=64):

        dataset_length = self.x_train.shape[0]

        index_start = self.train_index
        index_stop = ( self.train_index + batch_size ) % dataset_length

        if index_start < index_stop:
            # standard case:
            x = self.x_train[index_start:index_stop]
            y = self.y_train[index_start:index_stop]
        else:
            # special case where we wrap around back to the front again
            x = np.concatenate((self.x_train[index_start:],self.x_train[:index_stop]))
            y = np.concatenate((self.y_train[index_start:],self.y_train[:index_stop]))

        self.train_index = index_stop

        assert x.shape[0] == batch_size, f"x shape={x.shape[0]}, batch_size={batch_size}"
        assert y.shape[0] == batch_size, f"y shape={y.shape[0]}, batch_size={batch_size}"

        return (x, y)

    def get_test_batch(self, batch_size=64):

        dataset_length = self.x_test.shape[0]

        index_start = self.test_index
        index_stop = ( self.test_index + batch_size ) % dataset_length

        if index_start < index_stop:
            # standard case:
            x = self.x_test[index_start:index_stop]
            y = self.y_test[index_start:index_stop]
        else:
            # special case where we wrap around back to the front again
            x = np.concatenate((self.x_test[index_start:],self.x_test[:index_stop]))
            y = np.concatenate((self.y_test[index_start:],self.y_test[:index_stop]))

        self.test_index = index_stop

        assert x.shape[0] == batch_size, f"x shape={x.shape[0]}, batch_size={batch_size}"
        assert y.shape[0] == batch_size, f"y shape={y.shape[0]}, batch_size={batch_size}"

        return (x, y)

    def _test(self): # unit test

        loader = DataLoader()
        loader.generate_dataset()
        for _ in range(int(2.2*loader.x_train.shape[0])):
            loader.get_train_batch()
        for _ in range(int(2.2*loader.x_test.shape[0])):
            loader.get_test_batch()



