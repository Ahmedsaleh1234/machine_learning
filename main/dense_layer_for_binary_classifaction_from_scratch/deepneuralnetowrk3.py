import numpy as np

class deep():
    def __init__(self, nx, layers):
        self.__weights = {}
        self.__L = len(layers)
        self.__cache = {}
        for i in range(self.__L):
            wi = 'w' + str(i + 1)
            bi = 'b' + str(i + 1)
            if  i == 0:
                self.__weights[wi] = np.random.normal(size=(layers[i], nx)
                                                      ) * np.sqrt(2 / nx)
            else:
                self.__weights[wi] = np.random.normal(size=(layers[i],layers[i -1])
                                                      ) * np.sqrt(2 / layers[i- 1])
            self.__weights[bi] = np.zeros((layers[i], 1))
    
    def forward_prop(self, X):
        X = X.T
        self.__cache['A0'] = X 
        for i in range(self.__L):
            h = np.matmul(self.__weights['w' + str(i+1)], self.__cache['A' + str(i)]
                          ) + self.__weights['b' + str(i+1)]
            A = 1 / (1 +np.exp(-h))
            self.__cache['A' + str(i + 1)] = A
        return A, self.__cache
    
    def cost(self, A, Y):
        A = np.clip(A, 1e-15, 1-1e-15)
        m = Y.shape[1] 
        binary_cross_entropy_loss = (-1 / m) * np.sum(((1 - Y) * np.log(1 - A)
                                                       ) + (Y * np.log(A)))
        return binary_cross_entropy_loss
    
    def backward_prop(self, Y, cash, alpha=.05):
        dz = self.__cache['A' + str(self.__L)] - Y
        m = Y.shape[1]
        for i in range(self.__L, 0, -1):
            dw = (1/ m) * np.matmul( self.__cache['A' +str(i - 1)], dz.T)
            db = np.sum(dz, axis=1, keepdims=True)
            dz = (self.__cache['A' +str(i - 1)] * (1 - self.__cache['A' + str(i -1)])
                  ) * np.matmul(self.__weights['w' +str(i)].T, dz)
            self.__weights['b' + str(i)] = self.__weights['b' + str(i)] - db * alpha
            self.__weights['w' + str(i)] = self.__weights['w' + str(i)] - (dw * alpha).T

    @property
    def weights(self):
        return  self.__weights

            

        

