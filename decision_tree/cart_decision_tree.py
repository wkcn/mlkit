'''
CART决策树
对于连续特征，分为两份{x|x<=s}和{x|x>s}
对于离散特征，分为两份{x|x==s}和{x|x!=s}
'''
from collections import defaultdict

from collections import defaultdict
import numpy as np
import pandas
from decision_tree import read_csv, Gini


class Node:
    def __init__(self, name=None, s=None, continuous=False, cls=None):
        '''
        name: 特征名字
        s: 切分点
        cls: 叶子类别
        '''
        self.name = name

        if cls is None:
            self.leaf = False
            self.s = s
            self.continuous = continuous
            self.leaf_cls = None  # 叶子类别
            self.nodes = [None, None]
        else:
            self.leaf = True
            self.s = None
            self.leaf_cls = cls  # 叶子类别
            self.nodes = None
    def __str__(self):
        return self.get_str()
    def get_str(self, space=0):
        prefix = ' ' * space
        if self.leaf:
            leaf_str = f'predict: {self.leaf_cls}'
            return prefix + leaf_str
        if self.continuous:
            op_left, op_right = '<=', '>'
        else:
            op_left, op_right = '==', '!='
        s = prefix + f'{self.name} {op_left} {self.s}\n'
        s += self.nodes[0].get_str(space + 4) + '\n'
        s += prefix + f'{self.name} {op_right} {self.s}\n'
        s += self.nodes[1].get_str(space + 4) + '\n'
        return s


class CARTDecisionTree:
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
            return Node(cls=self._get_most_label(Y))
        ys = np.unique(Y)
        if len(ys) == 1:
            # 所有实例属于同一类
            return Node(cls=ys[0])
        best_gini = None
        best_iv = None
        for i in range(num_features):
            feature = X[:, i]
            fs = np.unique(feature)
            for v in fs:
                mask_eq = feature == v
                mask_ne = ~mask_eq
                g = sum(len(mask) / len(feature) * M.H(Y[mask]) for mask in [mask_eq, mask_ne])
                if best_gini is None or g < best_gini:
                    best_gini = g
                    best_iv = (i, v)
        best_i, best_v = best_iv
        feature = X[:, best_i]
        name = header[best_i]
        node = Node(name=name, s=best_v)
        mask = feature == best_v
        node.nodes[0] = self._get_node(
                np.delete(X[mask], best_i, axis=1),
                Y[mask],
                np.delete(header, best_i))
        node.nodes[1] = self._get_node(
                X[~mask],
                Y[~mask],
                header)
        return node
    def _get_most_label(self, labels):
        cnts = defaultdict(int)
        assert len(labels) == len(weights)
        for label, w in zip(labels, weights):
            cnts[label] += 1
        best = 0
        best_label = None
        for k, v in cnts.items():
            if v > best:
                best = v
                best_label = k
        return best_label

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
            node = node.nodes[x[i] != node.s]
        assert node.leaf
        return node.leaf_cls
    def __str__(self):
        return str(self.root)


if __name__ == '__main__':
    fname = '../data/table5.1.csv'
    columns, data = read_csv(fname)
    use_id = False
    '''
    只有C4.5能处理use_id = True
    '''
    offset = 0 if use_id else 1
    X, Y = data[:, offset:-1], data[:, -1]
    header = columns[offset:-1]
    dt = CARTDecisionTree()
    dt.train(X, Y, header)
    print(dt)
    PY = dt.predict(X, header)
    acc = (PY == Y).sum() / len(Y)
    print(f"Accuracy: {acc}")
