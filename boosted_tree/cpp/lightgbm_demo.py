import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt


def compute_acc(a, b):
    assert a.ndim == 1
    if a.shape != b.shape:
        return 0
    return np.count_nonzero((a >= 0.5) == (b >= 0.5)) / len(a)


def compute_rmse(a, b):
    return np.sqrt(np.square(a - b).mean())


def evaluate(preds, labels, prefix):
    acc = compute_acc(preds, labels)
    print(f"{prefix} Accuracy: {acc}")
    rmse = compute_rmse(preds, labels)
    print(f"{prefix} RMSE: {rmse}")


train_fname = './data/agaricus.txt.train'
test_fname = './data/agaricus.txt.test'

# read in data
dtrain = lgb.Dataset(train_fname, free_raw_data=True)
dtest = lgb.Dataset(test_fname, free_raw_data=True)

param = {'max_depth': 2, 'learning_rate': 1, 'objective': 'binary',
         'metric': ['binary_logloss', 'binary_error', 'rmse']}
num_round = 2
bst = lgb.train(param, dtrain, num_round, valid_sets=[dtest])

num_trees = bst.num_trees()
print(f'Number of trees: {num_trees}')
for i in range(num_trees):
    lgb.plot_tree(bst, tree_index=i)
plt.show()
