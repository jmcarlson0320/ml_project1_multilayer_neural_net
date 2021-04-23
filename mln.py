import numpy as np
from scipy.special import expit as sigmoid


'''
weights update algorithm using matrix operations
run the input vector through the perceptron
extract the predicted output class
if out not equal to target
    calculate the threshold of the output
    convert target to one_hot vector
    compute error vector
    wrap error in matrix
    transpose error vector
    wrap input in matrix
    matrix multiply error vector and input vector (gives a matrix)
    scale by learning rate
    add this delta matrix to the weights matrix
'''


class mln:

    # TODO setup a bias for the hidden layer
    # sets perceptron parameters and initializes weights matrix
    def __init__(self, num_input_nodes, num_output_nodes, num_hidden_nodes):
        self.num_in = num_input_nodes + 1
        self.num_out = num_output_nodes
        self.num_hidden = num_hidden_nodes
        self.hidden_weights = np.random.default_rng().uniform(-0.05, 0.05, (self.num_hidden, self.num_in))
        self.output_weights = np.random.default_rng().uniform(-0.05, 0.05, (self.num_out, self.num_hidden))
        self.hidden_activations = np.zeros(num_hidden_nodes)
        self.output_activations = np.zeros(num_output_nodes)


    # data = pre_process(data)
    def pre_process(self, input_data):
        # flatten each data point to a 1d array
        dim = np.shape(input_data)
        input_data = np.reshape(input_data, (dim[0], dim[1] * dim[2]))

        # scale each data point to be a value between 0 and 1
        scale = 1.0 / 255.0
        input_data = input_data * scale

        # add bias node to beginning of each data point
        bias = np.ones((dim[0], 1))
        input_data = np.concatenate((bias, input_data), axis=1)
        return input_data


    # TODO add a momentum term as a parameter
    def train(self, input_data, output_data, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            count = 1
            for data, target in zip(input_data, output_data):
                y = self.forward(data)
                prediction = np.argmax(y)
                if prediction != target:
                    # back propagation goes here...
                    # generate vector form of target: [0.1, 0.1, ... 0.9, ... 0.1]
                    t = np.zeros(self.num_out)
                    t.fill(0.1)
                    t[target] = 0.9
                    
                    # calculate output error using y(1 - y)(t - y)
                    a = np.add(1, -y)
                    b = np.add(t, -y)
                    ab = np.multiply(a, b)
                    err_out = np.multiply(y, ab)

                    # calculate hidden layer error using h(1 - h) * dot(transpose(hidden_weights), err_out)
                    a = np.add(1, -self.hidden_activations)
                    b = np.dot(np.transpose(self.output_weights), err_out)
                    ab = np.multiply(a, b)
                    err_hidden = np.multiply(self.hidden_activations, ab)

                    # update output weights using w = w + eta * err_out * hidden_activations
                    err_out = np.array([err_out])
                    err_out = np.transpose(err_out)
                    h = np.array([self.hidden_activations])
                    d_w_out = np.dot(err_out, h)
                    d_w_out = d_w_out * learning_rate
                    self.output_weights += d_w_out
    

                    # update hidden weights using w = w + eta * err_hidden * input
                    err_hidden = np.array([err_hidden])
                    err_hidden = np.transpose(err_hidden)
                    x = np.array([data])
                    d_w_hidden = np.dot(err_hidden, x)
                    d_w_hidden = d_w_hidden * learning_rate
                    self.hidden_weights += d_w_hidden

                print('\r# input data: ' + str(count), end = '')
                count += 1
            # calculate accuracy???
            # do other per-epoch things
        print('')


    def confusion_matrix(self, input_data, target_labels):
        cnf = np.zeros((10, 10))
        hidden_activations = sigmoid(np.dot(input_data, np.transpose(self.hidden_weights)))
        output_activations = sigmoid(np.dot(hidden_activations, np.transpose(self.output_weights)))
        for y, t in zip(output_activations, target_labels):
            val = np.argmax(y)
            cnf[val][t] += 1
        return cnf


    def accuracy(self, confusion_matrix):
        total = 0;
        correct = 0
        for i in range(self.num_out):
            for j in range(self.num_out):
                total += confusion_matrix[i][j]
                if i == j:
                    correct += confusion_matrix[i][j]
        return correct / total


    # returns output vector
    def forward(self, input_vector):
        self.hidden_activations = np.dot(self.hidden_weights, input_vector)
        self.hidden_activations = sigmoid(self.hidden_activations)
        self.output_activations = np.dot(self.output_weights, self.hidden_activations)
        self.output_activations = sigmoid(self.output_activations)
        return self.output_activations
