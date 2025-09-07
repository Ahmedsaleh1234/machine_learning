import numpy as np

class DeepNeuralNetwork():
    def __init__(self, nx, layers):
        self.weights = {}
        self.cache = {}
        self.L = len(layers)
        
        for i in range(self.L):
            w = 'w' + str(i + 1)
            b = 'b' + str(i + 1)
            if i == 0:
                self.weights[w] = np.random.randn(layers[i], nx)  * np.sqrt(2. / nx)
            else:
                self.weights[w] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2. /nx)
            self.weights[b] = np.zeros((layers[i], 1))