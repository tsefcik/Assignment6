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
        hidden_layer = [{"weights": [np.random.random() for index in range(inputs + 1)]} for index in
                        range(hidden_layers)]
        output_layer = [{"weights": [np.random.random() for index in range(hidden_layers + 1)]} for index in
                        range(possible_outputs)]

        neural_network.append(hidden_layer)
        neural_network.append(output_layer)

        return neural_network

    # Calculate activation by returning the sum of weights multiplied by the inputs
    def calc_activation(self, weights, inputs):
        # Get bias from weights list
        calculated_activation = weights[-1]

        # For every entry in weights, except for the bias, calculate the sum of the weight*input value
        for index in range(len(weights) - 1):
            # Compute activation function
            calculated_activation += float(weights[index]) * float(inputs[index])

        return calculated_activation

    # Forward propagate the network
    def forward_propagate_network(self, input_row, network):
        # Set row equal to variable
        input_to_work_with = input_row
        # Iterate through the neural network
        for layer in network:
            # Setup the outputs from the layer
            updated_input = []
            # Iterate through the layers
            for neuron in layer:
                # Get the calculated activation
                calculated_activation = self.calc_activation(weights=neuron["weights"], inputs=input_to_work_with)
                # Compute the value needed with the sigmoid function
                neuron["output"] = 1 / (1 + np.exp(-calculated_activation))
                updated_input.append(neuron["output"])

            input_to_work_with = updated_input

        return input_to_work_with

        # Backpropagation algorithm that will help train our feedforward neural network
    def backpropagation(self, network, output_pattern):
        # Get network length
        network_length = len(network)
        # Iterate through network backwards since we're doing backpropagation
        for index in reversed(range(network_length)):
            # Get current layer and length
            current_layer = network[index]
            current_layer_length = len(current_layer)
            # Keep track of errors that will be used later
            error_list = []

            # If it is the last layer in the list being iterated through backwards, get output error
            if index == len(network) - 1:
                # Iterate through each the layer
                for inner_index in range(current_layer_length):
                    # Get the neuron at this index
                    neuron = current_layer[inner_index]
                    # Computer error for this neuron
                    error = output_pattern[inner_index] * neuron["output"]
                    # Add to error list for current layer
                    error_list.append(error)

            # Otherwise get hidden layer error
            else:
                # Iterate through the current layer
                for inner_index in range(current_layer_length):
                    # Instantiate error associated here
                    error = float(0)
                    # Add error of each neuron at this layer together
                    for neuron in network[index + 1]:
                        error += neuron["weights"][inner_index] * neuron["delta"]
                    # Add error to error list for current layer
                    error_list.append(error)

            # Iterate through neurons at the current layer
            for inner_index in range(current_layer_length):
                # Get neuron
                neuron = current_layer[inner_index]
                # Computer delta for the current neuron
                delta = error_list[inner_index] * (neuron["output"] * (1 - neuron["output"]))
                if delta == 0:
                    delta = .01
                # Save that in the dictionary
                neuron["delta"] = delta

    def update_nn_weights(self, network, learning_rate, input_row):
        network_length = len(network)

        for layer in range(network_length):
            feature_values = input_row[:-1]

            if layer != 0:
                feature_values = []
                for neuron in network[layer-1]:
                    feature_values.append(neuron["output"])
            for neuron in network[layer]:
                fv_length = len(feature_values)
                for inner_index in range(fv_length):
                    neuron["weights"][inner_index] += learning_rate * neuron["delta"] * feature_values[inner_index]
                neuron["weights"][-1] += learning_rate * neuron["delta"]

    # Train the neural network given the network, data, learning rate, epoch, and number of outputs
    def train(self, network, training_data, learning_rate, epoch):
        # Iterate through number of cycles we specify
        for index in range(epoch):
            error = 0
            # Iterate through rows in the training data
            for inputs in training_data.values:
                # Forward propagate the network
                self.forward_propagate_network(input_row=inputs, network=network)
                expected_outputs = np.zeros(training_data.shape[0])
                expected_outputs[int(inputs[-1])] = 1

                # Perform backpropagation
                self.backpropagation(network=network, output_pattern=expected_outputs)
                # Update neural network weights
                self.update_nn_weights(network=network, learning_rate=learning_rate, input_row=inputs)

    # We just test our remaining data using our trained network by running forward propagation on each row
    def test(self, network, input_row):
        # Outputs from running through trained model
        outputs = self.forward_propagate_network(input_row=input_row, network=network)
        return outputs.index(max(outputs))
