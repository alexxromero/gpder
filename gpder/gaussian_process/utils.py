import numpy as np

def _atleast2d(arr):
    if np.ndim(arr) < 2:
        return np.array(arr).reshape(-1, 1)
    else:
        return np.array(arr)
