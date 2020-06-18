import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hog import get_hog_feat
from tqdm import tqdm

LR = 1e-2
batch_size = 10
num_label = 10
num_epoch = 3
USE_SOFTMAX = True 
NORMALIZE = True
USE_HOG = False

data = pd.read_csv('../data/mnist.csv', header=0).values

raw_X = data[:, 1:].astype(np.float32)
if USE_HOG:
    xs = []
    for i in range(len(raw_X)):
        im = raw_X[i].reshape((28, 28)).astype(np.uint8)
        feat = get_hog_feat(im)
        xs.append(feat.ravel())
    raw_X = np.array(xs, dtype=np.float32)
else:
    if NORMALIZE:
        raw_X_mean = raw_X.mean()
        raw_X_max = raw_X.max()
        raw_X_min = raw_X.min()
        raw_X = (raw_X - raw_X_mean) / (raw_X_max - raw_X_min)

raw_Y = data[:, 0] # [0, 9]

# one hot
Y = np.eye(num_label)[raw_Y.ravel()]

# -1, +1
if not USE_SOFTMAX:
    Y = Y * 2 - 1

Y = Y.astype(np.float32) 

train_X, test_X, train_Y, test_Y = train_test_split(raw_X, Y, test_size=0.33, random_state=42) 

train_dataset = gluon.data.dataset.ArrayDataset(train_X, train_Y)
test_dataset = gluon.data.dataset.ArrayDataset(test_X, test_Y)

train_loader = gluon.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = gluon.data.DataLoader(test_dataset, batch_size=batch_size)

class Perceptron:
    def __init__(self, lr, in_ndim, out_ndim):
        self.lr = lr
        self.W = mx.nd.random.normal(0, 0.01, shape = (out_ndim, in_ndim))
        self.b = mx.nd.zeros((out_ndim, 1))
    def train(self, x, y):
        '''
        x: (batch_size, in_ndim)
        y: (batch_size, out_ndim)
        W: (out_ndim, in_ndim)
        '''
        # (batch_size, out_ndim)
        pred = mx.nd.dot(x, self.W.T) + self.b.T
        # (batch_size, out_ndim)
        if not USE_SOFTMAX:
            wrong = (y * pred) < 0
            y_wrong = y * wrong
            self.W += self.lr * mx.nd.dot(y_wrong.T, x)
            self.b += self.lr * y_wrong.T.sum(1, keepdims=True)
        else:
            softmax = mx.nd.softmax(pred)
            # (batch_size, out_ndim)
            dY = softmax - y
            self.W -= self.lr * mx.nd.dot(dY.T, x)
            self.b -= self.lr * dY.T.sum(1, keepdims=True)
    def predict(self, x):
        pred = mx.nd.dot(x, self.W.T) + self.b.T
        return pred.argmax(1)

model = Perceptron(lr=LR, in_ndim=raw_X.shape[1], out_ndim=num_label)
print("Start Training...")
for epoch in range(1, num_epoch + 1):
    right = 0
    num_test = 0
    for x, y in tqdm(train_loader):
        model.train(x, y)
        pred = model.predict(x)
        label = y.argmax(1) 
        right += (pred == label).sum().asscalar()
        num_test += len(label)
    print ("Epoch %d [train]: %.2f" % (epoch, right * 1.0 / num_test)) 

    right = 0
    num_test = 0
    for x, y in test_loader:
        pred = model.predict(x)
        label = y.argmax(1) 
        right += (pred == label).sum().asscalar()
        num_test += len(label)
    print ("Epoch %d [test]: %.2f" % (epoch, right * 1.0 / num_test)) 

W = model.W.asnumpy()
for i in range(10):
    plt.subplot(4, 3, i + 1)
    plt.title('%d' % i)
    size = W[i].size
    L = int(np.sqrt(size))
    w = W[i].reshape((L, L))
    w_min = w.min()
    w_max = w.max()
    w = (w - w_min) / (w_max - w_min)
    plt.imshow(w, 'gray')
plt.show()
