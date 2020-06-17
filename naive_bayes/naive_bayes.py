import numpy as np
import pandas


def read_csv(fname):
    data = pandas.read_csv(fname)
    return np.array(data.to_numpy())


class NaiveBayes:
    def __init__(self):
        self.P_Y = None
        self.P_X_cond_Y = None
        self.num_features = None
    def train(self, X, Y):
        assert self.num_features is None
        assert X.ndim == 2
        assert Y.ndim == 1
        ys = np.unique(Y)
        self.P_Y = dict((y, np.count_nonzero(Y == y) / len(Y)) for y in ys)
        self.num_features = X.shape[1]
        self.P_X_cond_Y = [dict() for _ in range(self.num_features)]
        for j in range(self.num_features):
            Xj = X[:, j]
            xjs = np.unique(Xj)
            for xj in xjs:
                for y in ys:
                    mask = (Xj == xj) & (Y == y)
                    pair = (xj, y)
                    self.P_X_cond_Y[j][pair] = np.count_nonzero(mask) / np.count_nonzero(Y == y)
        print("Train over")
    def predict(self, X):
        Y = [None for _ in range(len(X))] 
        for i, x in enumerate(X):
            Y[i] = self.predict_one(x)
        return np.array(Y)
    def predict_one(self, x):
        num_features = len(x) 
        assert num_features == self.num_features
        names = list(self.P_Y.keys())
        probs = list()
        for y, py in self.P_Y.items(): 
            prob = py
            for j in range(num_features):
                pair = (x[j], y)
                prob *= self.P_X_cond_Y[j][pair]
            probs.append(prob)
        i = np.argmax(np.array(probs))
        return names[i] 


if __name__ == '__main__':
    fname = '../data/table4.1.csv'
    data = read_csv(fname)
    X, Y = data[:, :-1], data[:, -1]
    bayes = NaiveBayes()
    bayes.train(X, Y)
    PY = bayes.predict(X)
    acc = np.count_nonzero(Y == PY) / len(Y)
    print(f"Accuracy: {acc}")
    predict = bayes.predict_one([2, 'S'])
    print(f"input: [2, 'S'], predict: {predict}")
