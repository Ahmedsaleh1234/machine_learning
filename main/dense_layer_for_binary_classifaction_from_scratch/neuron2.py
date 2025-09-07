import numpy as np
class Neuron():
    def __init__(self, nx):
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W
    
    @property
    def b(self):
        return self.__b
    
    @property
    def A(self):
        return self.__A
    
    def frowrd_prop(self, x):
        x = x.T
        h = np.matmul(self.__W, x) + self.__b
        sigmoid = 1 / (1 + np.exp(-h))
        self.__A = sigmoid
        return self.__A
