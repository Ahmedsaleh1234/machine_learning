import numpy as np
import matplotlib.pyplot as plt
import pickle
class DeepNeuralNetwrok():
    def __init__(self, nx, layers, activation='sigmoid'):
        self.__weights = {}
        self.__L = len(layers)
        self.__cache = {}
        self._activation = activation

        for i in range(self.__L):
            wi = 'w' + str(i+ 1)
            bi = 'b' + str(i + 1)
            if i == 0:
                self.__weights[wi] = np.random.normal(size=(layers[i], nx)) * np.sqrt(2 /nx)
            else:
                self.__weights[wi] = np.random.normal(size=(layers[i], layers[i - 1] ))* np.sqrt(2/ layers[i -1])
            self.__weights[bi] = np.zeros((layers[i], 1))
    def forward_prop(self, X):
        X = X.T
        self.__cache['A0'] = X
        for i in range(self.__L):
            z = np.matmul(self.__weights['w'+str(i + 1)], self.__cache['A' +str(i)]
                          ) + self.__weights['b' +str(i + 1)]
            if i == self.__L:
                #softmax
                A = np.exp(z) / (np.sum(np.exp(z), axis=0, keepdims=True))
            else:
                #sigmoid
                if self._activation == 'sigmoid':
                    A = 1 / (1 + np.exp(-z))
                else:
                    #tanh
                    #A = (1 - np.exp(-2*z)) / (1 + np.exp(-2 * z))
                    A = np.tanh(z)
            self.__cache['A' + str(i+1)] = A
        return A, self.__cache
    def cost(self, Y, A):
        m = Y.shape[1]
        A = np.clip(A, 1e-15, 1- 1e-15)
        #binary_cross_enrtopy_loss = (-1 / m) * np.sum((((1-Y) * np.log(1-A)) + Y * np.log(A)))
        multiclass_cross_loss = (-1 / m) * np.sum(Y * np.log(A))
        return multiclass_cross_loss
    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= .5, 1, 0 ), cost
    
    def gradient_dectent(self, Y, alpha=.05):
        dz = self.__cache['A' + str(self.__L)] - Y
        m = Y.shape[1]
        for i in range(self.__L, 0, -1):
            dw = (1 / m) * np.matmul(self.__cache['A' + str(i-1)], dz.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            if self._activation == 'sigmoid':

                dz = np.matmul(self.__weights['w' + str(i)].T, dz) * ((1 - self.__cache['A' + str(i - 1)] ))\
                * self.__cache['A' + str(i - 1)]
            
            
            else:
                dz = np.matmul(self.__weights['w' + str(i)].T, dz) * (1 - self.__cache['A' + str(i -1)] ** 2)                                                                       
            self.__weights['w' + str(i)] = self.__weights['w' +str(i)] - (dw * alpha).T
            self.__weights['b' + str(i)] = self.__weights['b'+ str(i)] - (db * alpha)

    def train(self, X, Y, iterations=10, verbose=True, alpha=.05, graph=True, step=2):

        costs = []
        steps = list(range(0, iterations, step))
        for i in range(iterations):
            if verbose and i in steps:
                p, c = self.evaluate(X, Y)
                print(f'cost for iteration{i} is {c}')
                costs.append(c)
            self.forward_prop(X)
            self.gradient_dectent(Y, alpha)
        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel('iterations')
            plt.ylabel('costs')
            plt.suptitle('training')
            plt.show()
        return self.evaluate(X, Y)
    def save(self, filename):
        ''' Saves the instance object to a file in pickle format
            filename: file to which the object should be saved
        '''
        if '.pkl' not in filename:
            filename = filename + '.pkl'

        # open the file for writing
        with open(filename, 'wb') as fileObject:
            # this writes the object a to the file
            pickle.dump(self, fileObject)
    
    @staticmethod
    def load(filename):
        try:
            with open(filename, 'rb') as f:
                opj = pickle.load(f)
            return opj
        except FileNotFoundError:
            print('file not found')
            return None
                
        


