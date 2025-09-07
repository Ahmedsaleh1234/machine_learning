import numpy as np
from sensitivity import sensitivity
from pericision import precision
def f1score(confusion):
    recall = sensitivity(confusion)
    precision = precision(confusion)
    f1score = 2 * ((recall * precision) / (recall + precision))
    return f1score 

