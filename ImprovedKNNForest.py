import ID3
import pandas
import numpy as np
import random
import sklearn

class Tree:
    def __init__(self, solver: ID3.Node, centroid):
        self.solver = solver
        self.centroid = centroid


class KNNForest:
    def __init__(self, N, K, p, examples):

        self.trees = []
        self.n = examples.shape[0]
        self.centroids = np.zeros((N, examples.shape[1]))
        self.p = p
        self.N = N
        self.K = K

        for i in range(N):
            samples = random.sample(range(self.n), int(p * self.n))
            tree_examples = examples[samples]
            centroid = np.mean(tree_examples, axis=0)
            self.trees.append(Tree(ID3.ID3Solver(tree_examples, None).regularID3(m=10), centroid))
            self.centroids[i] = centroid

    def test(self, test_set=None):
        if test_set is None:
            test_set = np.array(pandas.read_csv("test.csv"))
            test_set[test_set[:, 0] == 'B', 0] = 1
            test_set[test_set[:, 0] == 'M', 0] = 0
        num_correct = 0
        for x in test_set:
            best_k_centroids = np.argsort(np.sum((self.centroids[:, 1:] - x[1:])**2, 1))[:self.K]
            deltas = np.sum((self.centroids[best_k_centroids][:, 1:] - x[1:])**2, 1)
            deltas[deltas < 0.0001] = 0.0001
            results = [0, 0]  # M, B
            for delta, i in zip(deltas, best_k_centroids):
                results[ID3.DT_class(x, self.trees[i].solver)] += 1/delta
            if results[0] >= results[1]:
                classification = 0
            else:
                classification = 1

            if classification == x[0]:
                num_correct += 1
        return num_correct / test_set.shape[0]


if __name__ == "__main__":
    E = np.array(pandas.read_csv("train.csv"))
    E[E[:, 0] == 'B', 0] = 1
    E[E[:, 0] == 'M', 0] = 0
    scalar = sklearn.preprocessing.MinMaxScaler()
    scalar.fit(E)
    E = np.array(scalar.transform(E))
    T = np.array(pandas.read_csv("test.csv"))
    T[T[:, 0] == 'B', 0] = 1
    T[T[:, 0] == 'M', 0] = 0
    T = scalar.transform(T)

    # run the following line to find the avg and max over 30 runs
    """   out = []
    for i in range(30):
        c = KNNForest(15, 11, 0.4, E)
        out.append(c.test(T))
    print(sum(out)/30)
    print(max(out))"""

    # this is for just one run
    c = KNNForest(15, 11, 0.4, E)
    print(c.test(T))
