import numpy as np

def sensitivity(confusion):
    true_pos = np.diag(confusion)
    false_neg = np.sum(confusion, axis=1) - true_pos

    sensitivity = true_pos / (true_pos + false_neg)
    return sensitivity