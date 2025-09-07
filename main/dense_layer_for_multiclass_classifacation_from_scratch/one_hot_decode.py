import numpy as np

def one_hot_decode(one_hot):
    return np.argmax(one_hot,axis=0)