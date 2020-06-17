from collections import defaultdict
from queue import Queue
import numpy as np

from decision_tree import DecisionTree, ID3, C4_5, read_csv

def prune(tree, X, Y, header, alpha):
    assert tree.root is not None
    parents = dict() # node to the parent of node
    q = Queue()
    q.put(tree.root)
    parents[tree.root] = None
    st = list()
    num_leaves = defaultdict(int) # the number of leaves
    while not q.empty():
        r = q.get()
        st.append(r)
        if not r.leaf:
            for node in r.nodes.values():
                assert node not in parents
                parents[node] = r
                q.put(node)
        else:
            num_leaves[node] = 1
    mheader = dict((name, i) for i, name in enumerate(header))
    samples = defaultdict(lambda: defaultdict(int)) # node to the label distribution of node
    losses = dict()
    for x, y in zip(X, Y):
        node = tree.root
        while not node.leaf:
            i = mheader[node.name]
            node = node.nodes[x[i]]
        assert node.leaf
        samples[node][y] += 1
    while st:
        r = st.pop() 
        # update its parent
        parent = parents[r] 
        samples_r = samples[r]
        ps = samples[parent]
        for k, v in samples_r.items():
            # update samples and num_leaves
            ps[k] += v
            num_leaves[parent] += num_leaves[r]

        t = np.array([v for v in samples_r.values()])
        n = sum(t)
        fr = t / n
        h = -(fr * np.log2(fr)).sum()
        loss = n * h + alpha * 1

        if r.leaf:
            losses[r] = loss
        else:
            losses[r] = sum(losses[node] for node in r.nodes.values())
            # try to prune
            if loss <= losses[r]:
                # prune
                r.leaf = True
                r.nodes = None
                best_cls = None
                best_cnt = None
                for cls, cnt in samples_r.items():
                    if best_cnt is None or cnt > best_cnt:
                        best_cnt = cnt
                        best_cls = cls
                r.leaf_cls = best_cls
                losses[r] = loss
    print(f"Prune Over, loss: {losses[tree.root]:.5}")

def compute_loss(tree, X, Y, header, alpha):
    mheader = dict((name, i) for i, name in enumerate(header))
    samples = defaultdict(lambda: defaultdict(int)) # node to the label distribution of node
    losses = dict()
    q = Queue()
    q.put(tree.root)
    for x, y in zip(X, Y):
        node = tree.root
        while not node.leaf:
            i = mheader[node.name]
            node = node.nodes[x[i]]
        assert node.leaf
        samples[node][y] += 1

    loss = 0
    while not q.empty():
        r = q.get()
        if not r.leaf:
            for node in r.nodes.values():
                q.put(node)
        else:
            # leaf
            loss += alpha
            samples_r = samples[r]

            t = np.array([v for v in samples_r.values()])
            n = sum(t)
            fr = t / n
            h = -(fr * np.log2(fr)).sum()
            loss += n * h
    return loss


if __name__ == '__main__':
    fname = '../data/table5.1.csv'
    columns, data = read_csv(fname)
    use_id = True
    offset = 0 if use_id else 1
    alpha = 8

    X, Y = data[:, offset:-1], data[:, -1]
    header = columns[offset:-1]
    # method = ID3
    method = C4_5
    dt = DecisionTree(method)
    dt.train(X, Y, header)
    loss = compute_loss(dt, X, Y, header, alpha)
    print(f"Before prune, loss: {loss:.5}")
    print(dt)

    prune(dt, X, Y, header, alpha=alpha)
    loss = compute_loss(dt, X, Y, header, alpha)
    print(f"After prune, loss: {loss:.5}")
    print(dt)

    PY = dt.predict(X, header)
    acc = (PY == Y).sum() / len(Y)
    print(f"Accuracy: {acc}")
