import numpy as np

def euc_dist(X1, X2): # pairwise distance
    distance = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            distance[i][j] = X1[i] - X2[j]
    return distance

def dist(X1, X2): # pairwise distance
    distance = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            distance[i][j] = X1[i] - X2[j]
    return distance

def euc_dist2(X1, X2): # pairwise distance
    distance = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            distance[i][j] = X1[i] - X2[j]
    return distance

def _atleast2d(arr):
    if np.ndim(arr) < 2:
        return np.array(arr).reshape(-1, 1)
    else:
        return np.array(arr)
