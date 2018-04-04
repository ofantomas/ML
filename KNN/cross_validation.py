import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds):
    arr = np.arange(n)
    np.random.shuffle(arr)
    arr_list = np.array_split(arr, n_folds)
    res = []
    for i in range(n_folds):
        indices = arr_list.copy()
        val = indices.pop(i)
        train = np.hstack(indices)
        res.append((train, val))
    return res


def cross_val_predict(ind, dist, classes, weights):
        if weights is True:
            weights = 1 / (10 ** (-5) + dist)
        else:
            weights = np.ones(ind.shape)
        res = np.zeros(ind.shape[0]) 
        max_dist = np.zeros(ind.shape[0])
        for i in classes:
            mask = ind == i
            temp = (mask * weights).sum(axis=1)
            res[max_dist < temp] = i
            max_dist = np.maximum(max_dist, temp)
        return res.astype(int)
            

def knn_cross_val_score(X, y, k_list, score='accuracy', cv=3, **kwargs):
    if cv is None:
        cv = kfold(len(X), 3)
    elif type(cv) == int:
        cv = kfold(len(X), cv)
    if score != 'accuracy':
        raise TypeError('Unknown score type')
    res = {}
    for k in k_list:
        res[k] = []
    cl = KNNClassifier(k=max(k_list), **kwargs)
    for i in range(len(cv)):
        cl.fit(X[cv[i][0]], y[cv[i][0]])
        dist, ind = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
        for k in k_list:
            pred = cross_val_predict(y[cv[i][0]][ind][::, :k:], dist[::, :k:], np.unique(y[cv[i][0]]),
                                     weights=cl.weights)
            res[k].append((pred == y[cv[i][1]]).sum() / len(y[cv[i][1]]))
    return res
