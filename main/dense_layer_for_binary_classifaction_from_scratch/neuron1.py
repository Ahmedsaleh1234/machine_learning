import numpy as np
class Neuron():
    def __init__(self, nx):
        self.__w = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def w(self):
        return self.__w
    @property
    def b(self):
        return self.__b
    @property
    def A(self):
        return self.__A
    