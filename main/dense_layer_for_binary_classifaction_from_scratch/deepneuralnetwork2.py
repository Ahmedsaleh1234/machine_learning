import numpy as np
import matplotlib.pyplot as plt
class DeepNeuralNetwork():
    def __init__(self, nx, layers):
        self.__weights = {}
        self.__cache = {}
        self.__L = len(layers)
        
        if type(layers) is not list:
            raise TypeError('layers must be list')
        for i in range(self.__L):
            w = 'w' + str(i + 1)
            b = 'b' + str(i + 1)
            if i == 0:
                self.__weights[w] = np.random.normal(size=(layers[i], nx)) * np.sqrt(2 / nx)
            else:
                self.__weights[w] = np.random.normal(size=(layers[i], layers[i -1])) * np.sqrt(2 / layers[i -1])

            self.__weights[b] = np.zeros((layers[i], 1))
    @property
    def weights(self):
        return self.__weights
    
    @property
    def L(self):
        return self.__L
    @property
    def cache(self):
        return self.__cache
    def forward_prop(self, X):
        X = X.T
        self.__cache['A0'] = X
        for i in range(self.__L):
            xi = self.__cache['A' + str(i)]
            h = np.matmul(self.__weights['w' + str(i + 1)], xi) + self.__weights['b'+ str(i+ 1)]
            A = 1 / (1 + np.exp(-h))
            self.__cache['A' + str(i+1)] = A
        return A, self.__cache
    def cost(self, Y, A):
        m = Y.shape[1]
        A = np.clip(A, 1e-15, 1-1e-15)
        binary_cost_entropy_loss = (-1/ m) * np.sum(((1- Y) * np.log(1 - A) + 
                                                     Y * np.log(A)))
        return binary_cost_entropy_loss
    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= .5, 1, 0 ), cost
    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        dz = self.__cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            dw = (1 / m) * np.matmul(dz, self.__cache['A' + str(i - 1)].T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dz = np.matmul(self.__weights['w' + str(i)].T, dz) * \
            (self.__cache['A' + str(i - 1)] * (1 - self.__cache['A' + str(i - 1)]))

            self.__weights['w'+ str(i)] = self.__weights['w' +str(i)] - (dw * alpha)
            self.__weights['b'+ str(i)] = self.__weights['b'+ str(i)] - (db * alpha)


    def train(self, X, Y, iterations=5000,alpha=.05, verbose= True, step=100, graph=True):
        steps = list(range(0, iterations, step))
        costs = []
        for i in range(iterations):
            if verbose and i in steps:
                p, c = self.evaluate(X, Y)
                costs.append(c)
                print(f'cost after iteration{i} : {c}')
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel('iterations')
            plt.ylabel('costs')
            plt.suptitle('traing cost')
            plt.show()
            

        return self.evaluate(X, Y)
            
