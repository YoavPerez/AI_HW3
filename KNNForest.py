import ID3
import pandas
import numpy as np
import random


class Tree:
    def __init__(self, solver: ID3.Node, centroid):
        self.solver = solver
        self.centroid = centroid


class KNNForest:
    def __init__(self, N, K, p):
        examples = np.array(pandas.read_csv("train.csv"))
        examples[examples[:, 0] == 'B', 0] = 1
        examples[examples[:, 0] == 'M', 0] = 0
        self.trees = []
        self.n = examples.shape[0]
        self.centroids = np.zeros((N, examples.shape[1]))
        self.p = p
        self.N = N
        self.K = K

        for i in range(N):
            sampels = random.sample(range(self.n), int(p * self.n))
            tree_examples = examples[sampels]
            centroid = np.mean(tree_examples, axis=0)
            self.trees.append(Tree(ID3.ID3Solver(tree_examples, None).regularID3(), centroid))
            self.centroids[i] = centroid

    def test(self):
        test_set = np.array(pandas.read_csv("test.csv"))
        test_set[test_set[:, 0] == 'B', 0] = 1
        test_set[test_set[:, 0] == 'M', 0] = 0

        num_correct = 0
        for x in test_set:
            best_k_centroids = np.argsort(np.sum((self.centroids - x)**2, 1))[:self.K]
            results = [0, 0]  # M, B
            for i in best_k_centroids:
                results[ID3.DT_class(x, self.trees[i].solver)] += 1
            if results[0] >= results[1]:
                classification = 0
            else:
                classification = 1

            if classification == x[0]:
                num_correct += 1
        return num_correct / test_set.shape[0]


if __name__ == "__main__":
    """ checking what is the best"""
    """for number_of_trees in range(6, 19, 2):
        for k_value in range(3, number_of_trees, 2):
            for p_value in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                avg = 0
                for i in range(5):
                    c = KNNForest(number_of_trees, k_value, p_value)
                    avg += c.test()
                print(number_of_trees, k_value, p_value, avg/5)"""
    avg = 0
    for i in range(5):
        c = KNNForest(16, 13, 0.35)
        avg += c.test()
    print(avg / 5)
