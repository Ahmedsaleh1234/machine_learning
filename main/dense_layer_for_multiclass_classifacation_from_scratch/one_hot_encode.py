import numpy as np


def oneHotEncode(Y, classes):
    one_hot = np.zeros((classes, len(Y)))
    for i in range(len(Y)):
        one_hot[Y[i]][i] = 1

    return one_hot
