import numpy as np
class Neuron():
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError('nx must be integer')
        if nx < 1:
            raise ValueError('nx must be postive')
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0        