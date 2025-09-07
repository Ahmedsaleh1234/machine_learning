import numpy as np

class Neuron():
    def __init__(self, nx):
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0
    @property
    def b(self):
        return self.__b
    @property
    def w(self):
        return self.__W
    
    @property
    def A(self):
        return self.__A
    
    def forward_prop(self, X):
        X = X.T
        h = np.matmul(self.__W, X)
        sigmoid = 1 / (1 + np.exp(-h))
        self.__A = sigmoid
        return self.__A
    
    def cost(self, Y, A):
        m = Y.shape[1]
        binary_cross_entropy_loss = -1 / m * np.sum(((1 - Y) * np.log(1 - A)) +
                                                    (Y * np.log(A)))
        return binary_cross_entropy_loss

     
    def back_prop(self, X, Y, A, alpha=.05):
        dz = A - Y
        m = Y.shape[1]
        dw = (1 / m) * np.matmul(dz,  X)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - dw * alpha
        self.__b = self.__b - db * alpha
    def predict(self, X, Y):
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        return np.where(self.__A >= .5, 1, 0), cost
    
    def train(self, X, Y, iterations=5000, alpha=.05):
        for iter in range(iterations):
            self.forward_prop(X)
            self.back_prop(X, Y, self.__A, alpha)
        return self.predict(X, Y)


    