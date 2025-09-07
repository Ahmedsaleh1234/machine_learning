import numpy as np

class Neuron():
    def __init__(self, nx):
        self.__b = 0
        self.__w = np.random.normal(size=(1, nx))
        self.__a = 0

    @property
    def b(self):
        return self.__b
    @property
    def w(self):
        return self.__w
    
    def forward_prop(self, x):
        x = x.T
        h = np.matmul(self.__w, x) + self.__b
        self.__a = 1 / (1 + np.exp(-h))
        return self.__a
    def cost(self, y, a):
        m = y.shape[1]
        binary_cross_entropy_loss = -1 / m * (np.sum(np.multiply(1 - y, np.log(1-a)) 
                                                                + np.multiply(y, np.log(a))))
        return binary_cross_entropy_loss
    def evaluate(self, x, y):
        self.forward_prop(x)
        return np.where(self.__a >= .5, 1 ,0)
    def gradient_descent(self, x, y, a, alpha=.05):
        m = x.shape[1]
        dz = a - y
        dw = (1 / m) * np.matmul(dz, x)
        db = (1 / m) * np.sum(dz)
        self.__w =  self.__w - dw * alpha
        self.__b = self.__b - db * alpha


