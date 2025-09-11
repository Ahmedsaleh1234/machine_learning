import numpy as np
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_filter = np.where((y_train == 0) | (y_train == 1))
test_filter = np.where((y_test == 0) | (y_test == 1))

x_train_binary = x_train[train_filter]
y_train_binary = y_train[train_filter]

x_test_binary = x_test[test_filter]
y_test_binary = y_test[test_filter]
np.savez('mnist_binary_01.npz', 
         x_train=x_train_binary, y_train=y_train_binary, 
         x_test=x_test_binary, y_test=y_test_binary)
