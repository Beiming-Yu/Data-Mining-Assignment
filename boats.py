import numpy as np
import traj_dist.distance as tdist
import pickle
import pandas as pd
import os

from sklearn.neighbors import KNeighborsClassifier

data_dir = '../data/archive/train_dataset/train'
Xs = []
ys = []
# for i in range(1, (18329 + 1) // 100):
for i in range(1, 110 + 1):
    data = pd.read_csv(os.path.join(data_dir, str(i) + '.csv'))
    lat = np.array(data.iloc[:, 1])
    lon = np.array(data.iloc[:, 2])
    X = np.vstack([lat, lon]).T
    y = data.iloc[:, -1][0]
    Xs.append(X)
    ys.append(y)

k = 3
# Xs = np.array(Xs, dtype='object')
# ys = np.array(ys)
X_train = Xs[:10]
X_test = Xs[100:102]
y_train = ys[:10]
y_test = ys[100:102]

cdist = tdist.cdist(X_test, X_train, metric="dtw")
idx = np.argpartition(cdist, k)[:, :k]
y_pred = []

for i in range(len(idx)):
    temp = [ys[j] for j in idx[i]]
    pred = max(set(temp), key=temp.count)
    y_pred.append(pred)
print(y_pred)
correct = np.sum([y_pred[i] == y_test[i] for i in range(len(y_test))])
acc = correct / len(y_test)
print(acc)
'''
knn = KNeighborsClassifier(n_neighbors=10, metric=tdist.dtw)
knn.fit(Xs[:100], ys[:100])
y_pred = knn.predict(Xs[100:])
'''
