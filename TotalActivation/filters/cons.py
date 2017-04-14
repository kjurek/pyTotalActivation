import numpy as np
from itertools import combinations


def cons(root):
    n = len(root)
    fil = np.zeros(n + 1, dtype=np.complex)
    fil[0] = 1
    for i in range(1, n + 1):
        fil[i] = np.power(-1.0, i) * np.sum(np.exp(np.sum(np.array(list(combinations(root, i))), 1)))
    return fil
