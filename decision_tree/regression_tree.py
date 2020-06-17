'''
最小二乘回归树
'''
from collections import defaultdict
import numpy as np
import pandas
from decision_tree import read_csv, Gini
from cart_decision_tree import Node


class RegressionTree:
    def __init__(self):
        self.root = None
    def train(self, X, Y, header):
        assert self.root is None
        self.root = self._get_node(X, Y, header)
    def _get_node(self, X, Y, header):
        M = Gini
        assert X.ndim == 2
        assert len(X) == len(Y)
        num_features = X.shape[1]
        if num_features == 0:
            # 特征集合为空
            return Node(cls=Y.mean())
        ys = np.unique(Y)
        if len(ys) == 1:
            # 所有实例属于同一类
            return Node(cls=ys[0])
        best_loss = None
        best_iv = None
        for i in range(num_features):
            feature = X[:, i]
            xs = np.sort(np.unique(feature))
            xs = (xs[1:] + xs[:-1]) / 2.0
            for v in xs:
                mask_eq = feature <= v
                mask_ne = ~mask_eq
                c1 = np.mean(Y[mask_eq])
                c2 = np.mean(Y[mask_ne])
                PY = np.empty_like(Y)
                PY[mask_eq] = c1
                PY[mask_ne] = c2
                loss = np.square(Y - PY).sum()
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_iv = (i, v)
        best_i, best_v = best_iv
        feature = X[:, best_i]
        name = header[best_i]
        node = Node(name=name, s=best_v, continuous=True)
        mask = feature < best_v
        node.nodes[0] = self._get_node(
                X[mask],
                Y[mask],
                header)
        node.nodes[1] = self._get_node(
                X[~mask],
                Y[~mask],
                header)
        return node
    def predict(self, X, header):
        mheader = dict((name, i) for i, name in enumerate(header))
        Y = np.empty((len(X), ), dtype=np.object)
        for i, x in enumerate(X):
            Y[i] = self._predict_one(x, mheader)
        return Y

    def _predict_one(self, x, mheader):
        node = self.root
        while not node.leaf:
            i = mheader[node.name]
            node = node.nodes[int(x[i] > node.s)]
        assert node.leaf
        return node.leaf_cls
    def __str__(self):
        return str(self.root)


if __name__ == '__main__':
    fname = '../data/table5.2.csv'
    header, data = read_csv(fname)
    X, Y = data[:, :-1], data[:, -1]
    rt = RegressionTree()
    rt.train(X, Y, header)
    print(rt)
    PY = rt.predict(X, header)
    loss = np.square(PY - Y).sum()
    print(Y, PY)
    print(f"Loss: {loss:.5}")
