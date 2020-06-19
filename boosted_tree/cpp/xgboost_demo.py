import numpy as np
import xgboost as xgb

def compute_acc(a, b):
    assert a.ndim == 1
    if a.shape != b.shape:
        return 0
    return np.count_nonzero((a > 0.5) & (b > 0.5)) / len(a)

# read in data
dtrain = xgb.DMatrix('./data/agaricus.txt.train')
dtest = xgb.DMatrix('./data/agaricus.txt.test')

# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction

train_preds = bst.predict(dtrain)
train_acc = compute_acc(train_preds, dtrain.get_label())
print(f"Training Accuracy: {train_acc}")
test_preds = bst.predict(dtest)
test_acc = compute_acc(test_preds, dtest.get_label())
print(f"Testing Accuracy: {test_acc}")
