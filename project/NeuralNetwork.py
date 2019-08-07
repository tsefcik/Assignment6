import numpy as np

"""
This class represents the neural network that will be used for training and testing the model.
"""


class NeuralNetwork:
    # Pass in data to train and test with
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    # Create a neural network with random values from 0 to 1
    # Pass in number of inputs (rows of data), number of hidden layers, and number of possible outputs
    def new_neural_network(self, inputs, hidden_layers, possible_outputs):
        neural_network = []
        hidden_layer = [{"weights": [np.random.random() for index in range(inputs + 1)]} for index in range(hidden_layers)]
        output_layer = [{"weights": [np.random.random() for index in range(hidden_layers + 1)]} for index in
                        range(possible_outputs)]

        neural_network.append(hidden_layer)
        neural_network.append(output_layer)

        print("Initial Neural Network:")
        for layer in neural_network:
            print(layer)

        return neural_network
