import numpy as np

def precision(confusion):
    true_pos = np.diag(confusion)

    false_neg = np.sum(confusion, axis=0) - true_pos

    precision = true_pos / (true_pos + false_neg)
    return precision
