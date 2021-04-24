import numpy as np


class RegressionNetwork(object):
    ''' Single hidden layered neural network that applies uses the sigmoid activation functions
        and returns predictions for regression purposes
    '''

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        # Save learning rate
        self.lr = learning_rate

        # define activation functions
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
            inputs:
                features: 2D array, each row is one data record, each column is a feature
                targets: 1D array of target values
        '''
        n_records = features.shape[0]

        # initialize delta change
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):

            # forward pass
            final_outputs, hidden_outputs = self.feed_forward(X)

            # backproagation
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        # updates weights
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def feed_forward(self, X):
        ''' 
            Performs a forward pass         
            inputs:
                X: features batch

        '''
        ### Forward Pass ###

        # calculate the hidden layer's output
        weights_to_hidden = self.weights_input_to_hidden
        hidden_inputs = np.dot(X, weights_to_hidden)
        hidden_outputs = self.sigmoid(hidden_inputs)

        # calculate the final output
        weights_to_output = self.weights_hidden_to_output
        final_inputs = np.dot(hidden_outputs, weights_to_output)
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Performs backpropagation and updates weights
            inputs:
                final_outputs: output from forward pass
                y: target (i.e. label) batch
                delta_weights_i_h: change in weights from input to hidden layers
                delta_weights_h_o: change in weights from hidden to output layers

        '''
        ### Backward pass ###

        # calculate the output error (difference between the ground truth and the prediction)
        error = y - final_outputs

        # calculate the output's error term
        output_error_term = error * 1.0

        # calculate the hidden layer's contribution to the error
        hidden_error = np.dot(
            output_error_term, self.weights_hidden_to_output.T)  # [:, None])

        # calculate the hidden layer's error term
        hidden_error_term = hidden_error * \
            hidden_outputs * (1 - hidden_outputs)

        # calculate the change between the inputs and the hidden layer
        delta_weights_i_h += hidden_error_term * X[:, None]

        # calculate the change between the hideen and the output
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
            inputs:
                delta_weights_i_h: change in weights from input to hidden layers
                delta_weights_h_o: change in weights from hidden to output layers
                n_records: number of records
        '''

        # apply gradient descent step and update weights
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features 
            input:
                features: 1D array of feature values
        '''

        weights_to_hidden = self.weights_input_to_hidden
        weights_to_output = self.weights_hidden_to_output

        # forward pass a set of data
        final_outputs, _ = feed_forward(features)

        return final_outputs
