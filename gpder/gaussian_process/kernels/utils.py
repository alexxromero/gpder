import numpy as np

def _atleast2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(-1, 1)
    elif arr.ndim == 1:
        return arr[:, np.newaxis]
    else:
        return arr
