import mln
import numpy as np
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt

# experiment parameters
num_hidden_nodes = 20
learning_rate = 0.1
momentum = 0.9
num_epochs = 50

# download the MNIST dataset for handwritten digits
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# init the perceptron and input data
p = mln.mln(784, 10, num_hidden_nodes)
x_train = p.pre_process(x_train)
x_test = p.pre_process(x_test)
print(np.shape(x_train))

# get an untrained baseline, should be about 10%
cnf = p.confusion_matrix(x_test, y_test)
initial_accuracy_on_test = p.accuracy(cnf)
print(f'initial accuracy on test data: {initial_accuracy_on_test}')

results = [] # accuracy calculations for plots
print('training perceptron...')
for i in range(num_epochs):
    print('epoch ' + str(i + 1) + ' of ' + str(num_epochs))
    p.train(x_train, y_train, learning_rate, momentum, 1)

    # calculate training data accuracy
    cnf = p.confusion_matrix(x_train, y_train)
    training_results = p.accuracy(cnf)

    # calculate test data accuracy
    cnf = p.confusion_matrix(x_test, y_test)
    test_results = p.accuracy(cnf)

    # save accuracy values for plot
    results.append((training_results, test_results))

    # print epoch summary
    print('accuracy on training set: ' + str(training_results))
    print('accuracy on test set: ' + str(test_results))
    print('')
    
    '''
    # quit when output settles
    if i > 0:
        diff = training_results - results[i - 1][0]
        if abs(diff) < 0.001:
            break
    '''

print('results:')

# pretty-print the confusion matrix
np.set_printoptions(suppress='True')
print(cnf)

# plot accuracy using matplot
plt.plot(results)
plt.show()
