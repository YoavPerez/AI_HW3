import numpy as np
import math
from math import log
import pandas
import sklearn.model_selection
import matplotlib.pyplot as plt

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
    num_m = examples[examples[:, 0] == 0].shape[0]
    num_b = examples[examples[:, 0] == 1].shape[0]
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

        if ig >= max_ig:  # choose the feature which gives best IG
            max_ig = ig
            f_max = f
            max_v = v
    return f_max, max_v


def TDIDT(examples: np.array, features, default, select, M, p):
    if examples.shape[0] <= M:
        return Node(None, None, [], default)

    c = majority_class(examples, p)

    if np.all(examples[:, 0] == c) or len(features) == 0:
        return Node(None, None, [], c)

    f, threshold = select(features, examples)
    subtrees = [(0, TDIDT(f(examples, threshold), features, c, select, M, p)),
                (1, TDIDT(f.rev_call(examples, threshold), features, c, select, M, p))]
    return Node(f, threshold, subtrees, c)


def DT_class(o, Tree: Node):
    f, c, children, threshold = Tree.f, Tree.c, Tree.children, Tree.threshold

    if children is None or len(children) == 0:
        return c
    for (val, sub_tree) in children:
        if f.single_call(o, threshold) == val:
            return DT_class(o, sub_tree)


def majority_class(examples: np.array, p=0.5):
    num_b = examples[examples[:, 0] == 0].shape[0]
    num_m = examples.shape[0] - num_b
    return 0 if num_b/examples.shape[0] > p else 1


def Cost_sensitive_id3(examples: np.array, features, M=0, p=0):
    c = majority_class(examples, p)
    return TDIDT(examples, features, c, MaxIG, M, p)


class CostSensitiveID3Solver:
    def __init__(self, train_set, test_set):
        self.E = np.array(pandas.read_csv(train_set))
        self.E[self.E[:, 0] == 'B', 0] = 1
        self.E[self.E[:, 0] == 'M', 0] = 0
        self.E = np.array(self.E, dtype=float)

        self.test = np.array(pandas.read_csv(test_set))
        self.test[self.test[:, 0] == 'B', 0] = 1
        self.test[self.test[:, 0] == 'M', 0] = 0

    def regularID3(self, E=None, m=0, p=0.5):
        E = self.E if E is None else E
        F = [Feature(i) for i in range(1, self.E.shape[1])]
        return Cost_sensitive_id3(E, F, m, p)

    def regularAcc(self, tree, test=None):
        test = self.test if test is None else test
        num_correct = 0
        for x in test:
            if DT_class(x, tree) == x[0]:
                num_correct += 1
        return num_correct / test.shape[0]

    def KFoldCostID3(self, P=None):
        if P is None:
            P = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        folds = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=207721481)
        out = []
        for p in P:
            acc = []
            for train_indices, test_indices in folds.split(self.E):
                decision_tree = self.regularID3(self.E[train_indices], 10, p)
                acc.append(self.loss_q_4(decision_tree, self.E[test_indices]))
            out.append(sum(acc) / len(acc))

        plt.plot(P, out, label='loss')
        plt.legend()
        plt.show()

        min_m = -1
        min_acc = 1000
        for m, acc in zip(P, out):
            if acc < min_acc:
                min_m = m
                min_acc = acc
        return min_m, min_acc
    def find_best_p(self, P=None):
        if P is None:
            P = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            out = []
            for p in P:
                decision_tree = self.regularID3(self.E, 10, p)
                out.append(self.loss_q_4(decision_tree, self.test))

            plt.plot(P, out, label='loss')
            plt.legend()
            plt.show()

            min_m = -1
            min_acc = 1000
            for m, acc in zip(P, out):
                if acc < min_acc:
                    min_m = m
                    min_acc = acc
            return min_m, min_acc

    def m_prune(self, m=0):
        return self.regularID3(m=m)

    def loss_q_4(self, tree, test=None):
        test = self.test if test is None else test
        loss = 0
        for x in test:
            if not DT_class(x, tree) == x[0]:
                if x[0] == 0:  # was ill and classified as healthy
                    loss += 1
                else: # wad healthy anf d classified as ill
                    loss += 0.1
        return loss / test.shape[0]


def experiment(id3_solver: CostSensitiveID3Solver, M=None):
    """this is function for q.3
        to use it:
        1) create classifier using: classifier = ID3Solver('train.csv', 'test.csv')
        2) call it with M - the m values you want. [10, 15, 20, 25, 30, 35, 40, 200] is default

        the function will plot a graph of the m-values and their accuracies

        @:param id3_solver: the classifier
        @:param M: values of m

        :return: (the best m found, the accuracy over the train, accuracy over the test set)
        """

    if M is None:
        M = [10, 15, 20, 25, 30, 35, 40, 200]
    min_m, min_acc = id3_solver.KFoldCostID3(M)
    acc_of_real_test = id3_solver.regularAcc(id3_solver.m_prune(min_m))
    return min_m, min_acc, acc_of_real_test


if __name__ == '__main__':
    classifier = CostSensitiveID3Solver('train.csv', 'test.csv')
    print(classifier.loss_q_4(classifier.regularID3(m=10, p=0.25)))
