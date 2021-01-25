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
            self.trees.append(Tree(ID3.ID3Solver(tree_examples, None).regularID3(), centroid))
            self.centroids[i] = centroid
    def KfoldKNNForest(self, examples):
        N = [5, 10, 15, 20, 25]
        K = [(1, 3, 5), (5, 7, 9), (9, 11, 13), (13, 15, 17), (19, 21, 23)]
        P = [0.3, 0.4, 0.5]
        folds = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=123456789)
        for n, k_list in zip(N, K):
            for k in k_list:
                for p in P:
                    for train_indices, test_indices in folds.split(examples):
                        acc = []
                        tree = KNNForest(n, k, p, examples[test_indices])
                        acc.append(tree.test())
                    print("N: ", n, "| K: ", k, "|p: ", p, "|accuracy: ", sum(acc)/len(acc))


    def test(self, test_set=None):
        if test_set is None:
            test_set = np.array(pandas.read_csv("test.csv"))
            test_set[test_set[:, 0] == 'B', 0] = 1
            test_set[test_set[:, 0] == 'M', 0] = 0

        num_correct = 0
        for x in test_set:
            best_k_centroids = np.argsort(np.sum((self.centroids[:, 1:] - x[:, 1:])**2, 1))[:self.K]
            results = [0, 0]  # M, B
            for i in best_k_centroids:
                deltas = [0]*self.K
                for i, center in enumerate(best_k_centroids):
                    deltas[i] = (center[])
                results[ID3.DT_class(x, self.trees[i].solver)] += 1
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
    c = KNNForest(15, 11, 0.4, E)
    print(c.test())
    #c.KfoldKNNForest(E)