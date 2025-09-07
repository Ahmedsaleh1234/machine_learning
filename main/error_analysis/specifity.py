import numpy as np

def specifity(confusion):
    true_pos = np.diag(confusion)
    false_pos = np.sum(confusion, axis=0) - true_pos
    false_neg = np.sum(confusion, axis=1) - true_pos
    true_neg = np.sum(confusion) - (true_pos + false_neg + false_pos)
    specifity = true_neg / (true_neg + false_pos)
    return specifity