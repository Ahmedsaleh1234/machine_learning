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
    
    def forward_prop(self, m):
        m = m.T
        h = np.matmul(self.__W, m) + self.__b
        self.__A = 1 / (1 + np.exp(-h))
        return self.__A
    def cost(self, Y, A):
        m = Y.shape[1]
        binary_cross_entropy_loss = -1 / m * (np.sum(np.multiply((1 - Y), np.log(1 - A))+ 
                                                     np.multiply(Y, np.log(A))))
        return binary_cross_entropy_loss
    
    def evaluate(self, x, y):
        self.forward_prop(x)
        cost = self.cost(y, self.__A)
        return np.where(self.__A >= .5, 1 ,0), cost