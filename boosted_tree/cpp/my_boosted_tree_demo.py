import numpy as np
import boosted_tree as bst


def compute_acc(a, b):
    assert a.ndim == 1
    if a.shape != b.shape:
        return 0
    return np.count_nonzero((a >= 0.5) == (b >= 0.5)) / len(a)


def compute_rmse(a, b):
    return np.sqrt(np.square(a - b).mean())


def evaluate(preds, labels, prefix):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = compute_acc(preds, labels)
    print(f"{prefix} Accuracy: {acc}")
    rmse = compute_rmse(preds, labels)
    print(f"{prefix} RMSE: {rmse}")


train_fname = 'data/agaricus.txt.train'
test_fname = 'data/agaricus.txt.test'

param = bst.BoostedTreeParam()
param.objective = 'binary:logistic'
param.learning_rate = 1
param.n_estimators = 2

model = bst.BoostedTree(param)

train_X, train_Y = bst.ReadLibSVMFile(train_fname)
test_X, test_Y = bst.ReadLibSVMFile(test_fname)

model.train(train_X, train_Y)

train_preds = model.predict(train_X)
evaluate(train_preds, train_Y, 'Training')

test_preds = model.predict(test_X)
evaluate(test_preds, test_Y, 'Testing')
