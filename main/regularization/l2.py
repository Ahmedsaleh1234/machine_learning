import numpy as np


def l2_reg_cost(cost, lambtha, wights, L, m):
    normal_sum = 0
    for i in range(1, L):
        wight = wights['W' + str(i)]
        normal_sum = normal_sum + np.sqrt(np.sum(np.square(wight)))
    l2 = cost + (lambtha / (2 * m) * normal_sum)
    return l2



