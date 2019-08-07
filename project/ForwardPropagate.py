import numpy as np
"""
This class handles the forward propagation of a neural network
"""


class ForwardPropagate:

    # Calculate activation by returning the sum of weights multiplied by the inputs
    def calc_activation(self, weights, inputs):
        # Get bias from weights list
        calculated_activation = weights[-1]

        # For every entry in weights, except for the bias, calculate the sum of the weight*input value
        for index in range(len(weights) - 1):
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

        print(input_to_work_with)
        return input_to_work_with
