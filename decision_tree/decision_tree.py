'''
决策树
ID3和C4.5
'''
from collections import defaultdict
import numpy as np
import pandas

def read_csv(fname):
    data = pandas.read_csv(fname)
    return data.columns, np.array(data.to_numpy())


class DecisionTreeMethod:
    @staticmethod
    def H(X):
        # 熵：随机变量不确定性的度量
        raise NotImplemented()
    @staticmethod
    def CondH(Y, X):
        # 条件熵：随机变量X给定的条件下随机变量Y的条件熵，H(Y|X)
        raise NotImplemented()
    @staticmethod
    def Gain(Y, X):
        # 信息增益
        raise NotImplemented()


class ID3(DecisionTreeMethod):
    @staticmethod
    def H(X):
        # 熵：随机变量不确定性的度量
        assert X.ndim == 1
        ux = np.unique(X)
        entropy = 0.0
        for x in ux:
            p = np.count_nonzero(X == x) / len(X)
            # 这里用以2为底的对数，loge也可以的
            entropy += - np.log2(p) * p if p != 0 else 0
        return entropy
    @staticmethod
    def CondH(Y, X):
        # 条件熵：随机变量X给定的条件下随机变量Y的条件熵，H(Y|X)
        assert Y.ndim == 1
        assert X.ndim == 1
        ux = np.unique(X)
        cond_entropy = 0
        for x in ux:
            mask = X == x
            p = np.count_nonzero(mask) / len(mask)
            ce = p * ID3.H(Y[mask])
            cond_entropy += ce
        return cond_entropy
    @staticmethod
    def Gain(Y, X):
        # 信息增益
        # g(D, A) = H(D) - H(D|A)
        return ID3.H(Y) - ID3.CondH(Y, X)


class C4_5(ID3):
    @staticmethod
    def Gain(Y, X):
        # 信息增益比
        return ID3.Gain(Y, X) / ID3.H(X)


class Gini(ID3):
    @staticmethod
    def H(X):
        # 熵：随机变量不确定性的度量
        assert X.ndim == 1
        ux = np.unique(X)
        p2sum = 0.0
        for x in ux:
            p = np.count_nonzero(X == x) / len(X)
            # 这里用以2为底的对数，loge也可以的
            p2sum += p * p
        return 1.0 - p2sum


class Node:
    def __init__(self, name=None, cls=None):
        '''
        name: 特征名字
        cls: 叶子类别
        '''
        self.name = name
        if cls is None:
            self.leaf = False
            self.leaf_cls = None  # 叶子类别
            self.nodes = dict()  # 特征名称 -> 节点
        else:
            self.leaf = True
            self.leaf_cls = cls  # 叶子类别
            self.nodes = None
    def __str__(self):
        return self.get_str()
    def get_str(self, space=0):
        prefix = ' ' * space
        if self.leaf:
            leaf_str = f'predict: {self.leaf_cls}'
            return prefix + leaf_str
        s = ''
        for name, node in self.nodes.items():
            s += prefix + f'{self.name} = {name}\n'
            s += node.get_str(space + 4) + '\n'
        return s


class DecisionTree:
    def __init__(self, method, gain_threshold=None):
        assert issubclass(method, DecisionTreeMethod)
        self.method = method
        self.root = None
        self.gain_threshold = gain_threshold
    def train(self, X, Y, header):
        assert self.root is None
        self.root = self._get_node(X, Y, header)
    def _get_node(self, X, Y, header):
        M = self.method
        '''
        print(M.H(Y))
        print(M.CondH(Y, X[:, 0]))
        print(M.Gain(Y, X[:, 0]))
        print(M.Gain(Y, X[:, 1]))
        print(M.Gain(Y, X[:, 2]))
        '''
        assert X.ndim == 2
        num_features = X.shape[1]
        if num_features == 0:
            # 特征集合为空
            return Node(cls=self._get_most_label(Y))
        ys = np.unique(Y)
        if len(ys) == 1:
            # 所有实例属于同一类
            return Node(cls=ys[0])
        # 选择信息增益最大的特征
        best_gain = None
        best_i = None
        for i in range(num_features):
            g = M.Gain(Y, X[:, i])
            if best_gain is None or g > best_gain:
                best_gain = g
                best_i = i

        if self.gain_threshold and best_gain < self.gain_threshold:
            # 信息增益小于阈值
            return Node(cls=self._get_most_label(Y))
        feature = X[:, best_i]
        name = header[best_i]
        node = Node(name=name)
        other_features = np.delete(X, best_i, axis=1)
        other_header = np.delete(header, best_i)
        uf = np.unique(feature)
        for v in uf:
            mask = feature == v
            xm = other_features[mask]
            ym = Y[mask]
            node.nodes[v] = self._get_node(xm, ym, other_header)
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
            node = node.nodes[x[i]]
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
    # method = ID3
    # method = C4_5
    method = Gini
    dt = DecisionTree(method)
    dt.train(X, Y, header)
    print(dt)
    PY = dt.predict(X, header)
    acc = (PY == Y).sum() / len(Y)
    print(f"Accuracy: {acc}")
