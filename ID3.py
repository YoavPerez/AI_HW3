import numpy as np
import math
from math import log
import pandas

""" the parameters are:
examples: numpy array of the rows in ths csv. [0] => classification, [i] => the ith feature
feature = i => the value of the example in column i
number of features = number of columns that are not the classification
features = 1, 2, 3, 4, ... , n (not including 0!!!)
"""

class Feature:
    def __init__(self, col):
        self.col = col

    def values(self, mat):
        v = sorted(set(mat[:, self.col]))
        return [0.5*(v[i] + v[i+1]) for i in range(len(v) - 1)]

    def __call__(self, mat, v):
        return mat[mat[:, self.col] <= v]


def H(examples):
    total = examples.shape[0]
    if examples.shape[0] == 0:
        return 0
    num_b = examples[examples[:, 0] == 0].shape[0]
    num_m = examples.shape[0] - num_b
    p_b = num_b / total
    p_m = num_m / total

    out = 0
    out -= 0 if num_b == 0 else log(p_b, 2)*p_b
    out -= 0 if num_m == 0 else log(p_m, 2) * p_m
    return out


def IG(f: Feature, mat: np.array):
    total = mat.shape[0]
    h = H(mat)
    for v in f.values(mat):
        ei = f(mat, v)


    max_val = -np.inf

    return max_val



def MaxIG(F, E):
    f_max = None
    max_val = -np.inf
    for f in F:
        val = IG(f, E)
        if val > max_val:
            max_val = val
            f_max = f
    return f_max

def TDIDT(E: np.array, F, default, select):
    if E.shape[0] == 0:
        return None, np.array([]), default

    c = majority_class(E)
    if np.all(E[:, 0] == c) or len(F) == 0:
        return None, np.array([]), c
    f = select(F, E)
    print(E.shape)
    subtrees = [(v, TDIDT(f(E, v), F, c, select)) for v in f.values(E)]

    return f, subtrees, c

def majority_class(E:np.array):
    num_b = E[E[:, 0] == 0].shape[0]
    num_m = E.shape[0] - num_b
    return 0 if num_b >= num_m else 1

def ID3(E:np.array, F):
    c = majority_class(E)
    return TDIDT(E, F, c, MaxIG)


if __name__ == '__main__':
    E = np.array(pandas.read_csv('train.csv'))
    E[E[:, 0] == 'B', 0] = 0
    E[E[:, 0] == 'M', 0] = 1
    E = np.array(E, dtype=float)
    F = [Feature(i) for i in range(1, E.shape[1])]
    tree = ID3(E, F)
    print(tree[0])
    quit()
