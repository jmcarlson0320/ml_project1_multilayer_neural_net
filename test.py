import mln
import numpy as np
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# init the perceptron and input data
p = mln.mln(784, 10, 20)
x_train = p.pre_process(x_train)
x_test = p.pre_process(x_test)
print(np.shape(x_train))

y = p.forward(x_train[0])

print(y)

