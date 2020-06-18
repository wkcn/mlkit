import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

n_samples = 300
centers = [[-10, -10], [0, 0], [3, 5]]
cluster_std = [0.2, 0.1, 0.3]
n_clusters = len(centers)
random_state = 3

X, Y = make_blobs(n_samples=n_samples, n_features=2,
                  centers=centers, cluster_std=cluster_std,
                  random_state=random_state)

R, C = X.shape
with open('data.txt', 'w') as fout:
    fout.write(f"{R} {C}\n")
    for r in range(R):
        fout.write(' '.join(f"{X[r][c]}" for c in range(C)) + '\n')

'''
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()
'''

res = KMeans(n_clusters=n_clusters, random_state=9).fit(X)
print(res.cluster_centers_, res.labels_, res.inertia_)

centers = res.cluster_centers_
tot_dis = 0.0
for i, c in enumerate(res.labels_):
    x = X[i]
    center = centers[c]
    diff = x - center
    dis = ((diff * diff).sum())
    tot_dis += dis
print(tot_dis)
