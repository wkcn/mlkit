'''
Adaboost

注意_get_most_label的实现，需要考虑样本的权重（样本的分布）
注意变量，用best_xxx
'''
from collections import defaultdict
import numpy as np
import pandas

def read_csv(fname):
    data = pandas.read_csv(fname)
    return np.array(data.to_numpy())


class Node:
    def __init__(self, s, c1, c2, am):
        self.s = s
        self.c1 = c1
        self.c2 = c2
        self.am = am
    def __str__(self):
        return f'[am: {self.am:.5}] {{x| x <= {self.s}}}: {self.c1}, {{x| x > {self.s}}}: {self.c2}'


class AdaBoost:
    def __init__(self):
        self.nodes = list()
    def train(self, X, Y):
        assert len(self.nodes) == 0
        n = len(X)
        w = np.ones(n) / n
        xs = np.sort(np.unique(X))
        xs = (xs[1:] + xs[:-1]) / 2.0
        it = 0
        while 1:
            min_err = None
            best_PY = None
            best_s = None
            best_c = None
            for i, s in enumerate(xs):
                R1 = X <= s
                R2 = X > s
                c1 = self._get_most_label(Y[R1], w[R1])
                c2 = self._get_most_label(Y[R2], w[R2])
                PY = np.empty_like(Y)
                PY[R1] = c1
                PY[R2] = c2
                err = (w * (PY != Y)).sum()
                print("DIVIDE: ", s, err, (PY!=Y))
                if min_err is None or err < min_err:
                    min_err = err
                    best_PY = PY
                    best_s = s
                    best_c = (c1, c2)
            print(f"[IT {it}] error: {min_err:.5}")
            am = 0.5 * np.log((1 - min_err) / min_err)
            node = Node(s=best_s, c1=best_c[0], c2=best_c[1], am=am)
            print(node)
            self.nodes.append(node)
            PY = self.predict(X)
            EY = (PY == Y)
            if EY.all():
                print("Train over")
                break
            else:
                print("Error Sample: {}".format(np.count_nonzero(~EY)))
            new_w = w * np.exp(-am * Y * best_PY)
            new_w /= np.sum(new_w)
            w = new_w
            print("Sample Weight:", w)
            it += 1
    def _get_most_label(self, labels, weights):
        # 因为权重改变了样本的分布，所以label也要受权重影响
        cnts = defaultdict(int)
        assert len(labels) == len(weights)
        for label, w in zip(labels, weights):
            cnts[label] += w
        best = 0
        best_label = None
        for k, v in cnts.items():
            if v > best:
                best = v
                best_label = k
        return best_label
    def predict(self, X):
        Y = np.zeros((len(X),), dtype=np.float32)
        for node in self.nodes:
            T = (X <= node.s).astype(np.float32)
            Y += node.am * (T * node.c1 + (1 - T) * node.c2)
        return (Y > 0) * 2 - 1
    def __str__(self):
        gs = ' + '.join(f'{node.am:.5}' + ' * {' + str(node) + '}\n' for node in self.nodes)
        s = f'G(x) = sign[{gs}]'
        return s

if __name__ == '__main__':
    fname = '../data/table8.1.csv'
    data = read_csv(fname)
    X, Y = data[:, 0], data[:, 1]
    ab = AdaBoost()
    ab.train(X, Y)
    print(ab)
