import numpy as np
def confusion_matraix(labels, loagisic):
    matrix = np.matmul(labels.T, loagisic)
    return matrix

