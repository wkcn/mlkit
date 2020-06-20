import numpy as np
import xgboost as xgb

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

# read in data
dtrain = xgb.DMatrix('./data/agaricus.txt.train')
dtest = xgb.DMatrix('./data/agaricus.txt.test')

# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
watch_list = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, watch_list)

# make prediction
train_preds = bst.predict(dtrain)
evaluate(train_preds, dtrain.get_label(), 'Training')
test_preds = bst.predict(dtest)
evaluate(test_preds, dtest.get_label(), 'Testing')
