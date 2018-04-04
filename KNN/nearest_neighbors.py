import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


def calculate_distance_euclidian(X, Y):
    return np.sqrt((-(2 * X.dot(Y.T))) + (X ** 2).sum(axis=1)[:, np.newaxis] + (Y.T ** 2).sum(axis=0)[np.newaxis, :])


def calculate_distance_cosine(X, Y):
    return 1 - X.dot(Y.T) / np.sqrt(((X ** 2).sum(axis=1)[:, np.newaxis]) * ((Y.T ** 2).sum(axis=0)[np.newaxis, :]))


class KNNClassifier:
    def __init__(self, k, strategy='my_own', metric='euclidean', weights=False, test_block_size=0):
        self.k = k
        if strategy == 'my_own':
            self.strategy = 'my_own'
        elif strategy == 'brute':
            self.strategy = 'brute'
        elif strategy == 'kd_tree':
            self.strategy = 'kd_tree'
        elif strategy == 'ball_tree':
            self.strategy = 'ball_tree'
        else:
            raise TypeError('Unknown strategy type')
        if metric == 'euclidean':
            self.metric = 'euclidean'
            self.metric_function = calculate_distance_euclidian
        elif metric == 'cosine':
            self.metric = 'cosine'
            self.metric_function = calculate_distance_cosine
        else:
            raise TypeError('Unknown metric type')
        if weights is True or weights is False:
            self.weights = weights
        else: 
            raise TypeError('Incorrect weights value')
        self.test_block_size = test_block_size
            
    def fit(self, X, y):
        if self.strategy == 'my_own':
            self.X = X
            self.y = y
        else:
            self.y = y
            self.neighbors_class = NearestNeighbors(self.k, algorithm=self.strategy, metric=self.metric)
            self.neighbors_class.fit(X, y)
    
    def find_kneighbors(self, X, return_distance=True):
        if self.strategy == 'my_own':
            if self.test_block_size == 0:
                distance = self.metric_function(self.X, X)
                index = np.argpartition(distance, range(self.k), axis=0)[:self.k:]
                distance = distance[index, np.arange(distance.shape[1])]
                if return_distance is True:
                    return tuple([distance.T, index.T])
                else:
                    return index.T
            else:
                ind = np.empty((0, self.k))
                dist = np.empty((0, self.k))
                for i in range(X.shape[0] // self.test_block_size):
                    distance = self.metric_function(self.X, X[i * self.test_block_size:(i + 1) * self.test_block_size:])
                    index = np.argpartition(distance, range(self.k), axis=0)[:self.k:]
                    distance = distance[index, np.arange(distance.shape[1])]
                    ind = np.vstack((ind, index.T))
                    dist = np.vstack((dist, distance.T)) 
                if X.shape[0] % self.test_block_size != 0:   
                    distance = self.metric_function(self.X, X[(X.shape[0]\
                     // self.test_block_size) * self.test_block_size:])
                    index = np.argpartition(distance, range(self.k), axis=0)[:self.k:]
                    distance = distance[index, np.arange(distance.shape[1])]
                    ind = np.vstack((ind, index.T))
                    dist = np.vstack((dist, distance.T))
                if return_distance is True:
                    return tuple([dist, ind.astype(int)])
                else:
                    return ind.astype(int)
        else:
            if self.test_block_size == 0:
                return self.neighbors_class.kneighbors(X, self.k, return_distance=return_distance)
            else:
                ind = np.empty((0, self.k))
                dist = np.empty((0, self.k))
                for i in range(X.shape[0] // self.test_block_size):
                    distance, index = self.neighbors_class.kneighbors(X[i * self.test_block_size:(i + 1) \
                     * self.test_block_size:], self.k, return_distance=True)
                    ind = np.vstack((ind, index))
                    dist = np.vstack((dist, distance))
                if X.shape[0] % self.test_block_size != 0:
                    distance, index = self.neighbors_class.kneighbors(X[(X.shape[0] \
                        // self.test_block_size) * self.test_block_size:], self.k, return_distance=True)
                    ind = np.vstack((ind, index))
                    dist = np.vstack((dist, distance))
                if return_distance is True:
                    return tuple([dist, ind.astype(int)])
                else:
                    return ind.astype(int)

    def predict(self, X):
        if self.weights is True:
            dist, ind = self.find_kneighbors(X)
            ind = self.y[ind]
            weights = 1 / (10 ** (-5) + dist)
            res = np.zeros(ind.shape[0])
            max_dist = np.zeros(ind.shape[0])
            for i in np.unique(self.y):
                mask = ind == i
                temp = (mask * weights).sum(axis=1)
                res[max_dist < temp] = i
                max_dist = np.maximum(max_dist, temp)
            return res.astype(int)
        else:
            ind = self.find_kneighbors(X, return_distance=False)
            ind = self.y[ind]
            res = np.zeros(ind.shape[0]) 
            max_dist = np.zeros(ind.shape[0])
            for i in np.unique(self.y):
                mask = ind == i
                temp = mask.sum(axis=1)
                res[max_dist < temp] = i
                max_dist = np.maximum(max_dist, temp)
            return res.astype(int)
