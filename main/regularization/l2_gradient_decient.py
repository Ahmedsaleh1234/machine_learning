import numpy as np 
def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    m = Y.shape[1]
    dz = cache['A' +str(L)] - Y
    for i in range(L, 0, -1):
        l2 = (lambtha / m) * np.sum(weights['W' +str(i)])
        dw = (1 / m) * np.matmul(dz, cache['A' + str(i - 1)].T) + l2
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = (1 - (cache['A' +str(i - 1)] ** 2)) * np.matmul(weights['W' +str(i)].T, dz)
        weights['W' + str(i)] = weights['W' +str(i)] - (dw * alpha)
        weights['b'+ str(i)] = weights['b' +str(i)] - (db * alpha)



