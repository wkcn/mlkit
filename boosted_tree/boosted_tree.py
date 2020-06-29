from collections import defaultdict
import numpy as np
import pandas

class Node:
    def __init__(self, s, c1, c2):
        self.s = s
        self.c1 = c1
        self.c2 = c2
    def __str__(self):
        return f'{{x| x <= {self.s}}}: {self.c1:.3}, {{x| x > {self.s}}}: {self.c2:.3}'

class BoostedTree:
    def __init__(self, threshold):
        self.threshold = threshold
        self.nodes = []
    def train(self, X, Y):
        assert len(self.nodes) == 0
        assert X.ndim == 1
        xs = np.sort(np.unique(X))
        xs = (xs[1:] + xs[:-1]) / 2.0
        it = 0
        RY = Y
        while 1:
            best_s = None
            best_c1 = best_c2 = None
            best_m = None

            for i, s in enumerate(xs):
                R1 = X <= s
                R2 = X > s
                Y1, Y2 = RY[R1], RY[R2]
                c1 = np.mean(Y1)
                c2 = np.mean(Y2)
                # loss
                ms = np.square(Y1 - c1).sum() + np.square(Y2 - c2).sum()
                if best_m is None or best_m > ms: 
                    best_m = ms
                    best_s = s
                    best_c1, best_c2 = c1, c2
            node = Node(s=float(best_s),
                    c1=float(best_c1),
                    c2=float(best_c2))
            self.nodes.append(node)
            predict_y = self.predict(X)
            loss = np.square(Y - predict_y).sum()
            print(f"[Iteration{it}] Loss: {loss:.3}")
            it += 1
            if loss < self.threshold:
                print("Train Over")
                break
            RY = Y - predict_y
    def predict(self, X):
        Y = np.zeros((len(X),), dtype=np.float32)
        for node in self.nodes:
            T = (X <= node.s).astype(np.float32)
            Y += T * node.c1 + (1 - T) * node.c2 
        return Y
    def __str__(self):
        s2node = defaultdict(list) 
        for node in self.nodes:
            s2node[node.s].append((node.c1, node.c2))
        nodes = []
        for s, c in s2node.items():
            c1 = sum([t[0] for t in c])
            c2 = sum([t[1] for t in c])
            nodes.append((s, c1, c2))
        nodes.sort(key=lambda x: x[0])
        v = sum(t[1] for t in nodes)
        left = nodes[0][0]
        s = f'{v}, x <= {nodes[0][0]}\n'
        v = v - nodes[0][1] + nodes[0][2]
        for node in nodes[1:]:
            right = node[0]
            s += f'{v}, {left} < x <= {right}\n'
            v = v - node[1] + node[2]
            left = right
        s += f'{v}, x > {left}\n'
        return s

def read_csv(fname):
    data = pandas.read_csv(fname)
    return np.array(data.to_numpy())

if __name__ == '__main__':
    data_fname = '../data/table8.2.csv'
    data = read_csv(data_fname)
    X, Y = data[:, 0], data[:, 1]

    bst = BoostedTree(threshold=0.2)
    bst.train(X, Y)
    print(bst)
