import numpy as np
import math
from math import log
import pandas
import time
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
        v = sorted(list(set(mat[:, self.col])))
        return [0.5 * (v[i] + v[i + 1]) for i in range(len(v) - 1)]

    def __call__(self, mat, v):
        out = mat[mat[:, self.col] <= v]
        return out

    def single_call(self, o, v):
        return 0 if o[self.col] <= v else 1

    def rev_call(self, mat, v):
        return mat[mat[:, self.col] > v]

class Node:
    def __init__(self, f, threshold, children, c):
        self.threshold = threshold
        self.children = children
        self.f = f
        self.c = c
def H(examples):
    total = examples.shape[0]
    if examples.shape[0] == 0:
        return 0
    num_b = examples[examples[:, 0] == 0].shape[0]
    num_m = examples[examples[:, 0] == 1].shape[0]
    p_b = num_b / total
    p_m = num_m / total

    assert num_b + num_m == total
    if num_b == 0:
        out = - log(p_m, 2) * p_m

    elif num_m == 0:
        out = -log(p_b, 2) * p_b
    else:
        out = (-log(p_b, 2) * p_b) + (- log(p_m, 2) * p_m)
    return out

def IG(f: Feature, mat: np.array):
    total = mat.shape[0]
    h = H(mat)
    max_v = -np.inf
    max_ig = -np.inf
    for v in f.values(mat):  # max IG for this feature
        ig = h
        e0 = f(mat, v)
        e1 = f.rev_call(mat, v)
        ig -= (((e0.shape[0] / total) * H(e0)) + ((e1.shape[0] / total) * H(e1)))
        if ig > max_ig:
            max_ig = ig
            max_v = v

    return max_ig, max_v


def MaxIG(features, examples):
    f_max = None
    max_ig = -np.inf
    max_v = -np.inf
    for f in features:  # max IG over all the features
        ig, v = IG(f, examples)

        if ig > max_ig:  # choose the feature which gives best IG
            max_ig = ig
            f_max = f
            max_v = v
    return f_max, max_v


def TDIDT(examples: np.array, features, default, select):
    if examples.shape[0] == 0:
        return Node(None, None, [], default)

    c = majority_class(examples)
    if np.all(examples[:, 0] == c) or len(features) == 0:
        return Node(None,None,  [], c)

    f, threshold = select(features, examples)
    subtrees = [(0, TDIDT(f(examples, threshold), features, c, select)),
                (1, TDIDT(f.rev_call(examples, threshold), features, c, select))]
    return Node(f, threshold, subtrees, c)


def DT_class(o, Tree:Node):
    f, c, children, threshold = Tree.f, Tree.c, Tree.children, Tree.threshold

    if children is None or len(children) == 0:
        return c
    for (val, sub_tree) in children:
        if f.single_call(o, threshold) == val:
            return DT_class(o, sub_tree)


def majority_class(examples: np.array):
    num_b = examples[examples[:, 0] == 0].shape[0]
    num_m = examples[examples[:, 0] == 1].shape[0]
    assert num_b + num_m == examples.shape[0]
    return 0 if num_b > num_m else 1


def ID3(examples: np.array, features):
    c = majority_class(examples)
    return TDIDT(examples, features, c, MaxIG)


if __name__ == '__main__':
    start = time.time()


    E = np.array(pandas.read_csv('train.csv'))
    E[E[:, 0] == 'B', 0] = 0
    E[E[:, 0] == 'M', 0] = 1
    E = np.array(E, dtype=float)
    F = [Feature(i) for i in range(1, E.shape[1])]
    tree = ID3(E, F)

    test = np.array(pandas.read_csv('test.csv'))
    test[test[:, 0] == 'B', 0] = 0
    test[test[:, 0] == 'M', 0] = 1

    num_correct = 0
    for x in test:
        if DT_class(x, tree) == x[0]:
            num_correct += 1
    print(num_correct / test.shape[0])
    print(time.time()-start)

