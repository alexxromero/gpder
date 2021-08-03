import numpy as np
import scipy
from scipy._lib._util import MapWrapper
import sklearn
from sklearn.utils.validation import check_random_state

__all__ = ['brute_minimizer', 'random_minimizer', 'hybrid_minimizer']

def brute_minimizer(fun, bounds, N=10, args=(), workers=1):
    """Grid search minimizer.

    Arguments
    ---------
    fun: callable
        Objective function to minimize.
    N: int
        No. of grid points along the axes.
    bounds: tuple
        Tuple where each element has of the form
        '(low, high)' with the corresponding parameter
        ranges.
    args: tuple, optional
        Additional fixed arguments of the objective
        function.
    workers: int, default=1
        If > 1, the grid is subdivided into 'workers'
        sections and evaluated in parallel.

    Returns
    -------
    xmin: ndarray
        Coordinates of the point which minimizes
        the objective function.
    fval: float
        Value of the objective function at 'xmin'.
    """
    nparams = len(bounds)
    lbounds = list(bounds)
    for i in range(nparams):
        step = (lbounds[i][1] - lbounds[i][0]) / N
        lbounds[i] = slice(lbounds[i][0], lbounds[i][1]+step, step)

    if (nparams == 1):
        lbounds = lbounds[0]

    grid = np.mgrid[lbounds]
    # reshape the grid to get the sets of nparams grid points
    mgrid_shape = np.shape(grid)
    if (nparams > 1):
        grid = np.reshape(grid, (mgrid_shape[0], np.prod(mgrid_shape[1:]))).T

    wrapped_fun = _Fun_Wrapper(fun, args)

    with MapWrapper(pool=workers) as mapper:
        fvals_grid = np.array(list(mapper(wrapped_fun, grid)))
        if (nparams == 1):
            grid = (grid,)
            fvals_grid = np.squeeze(fvals_grid)
        elif (nparams > 1):
            grid = np.reshape(grid.T, mgrid_shape)
            fvals_grid = np.reshape(fvals_grid, mgrid_shape[1:])

    fvals_shape = np.shape(fvals_grid)

    indx = np.argmin(fvals_grid.ravel(), axis=-1)
    # find the index corresponding to the min. value
    xmin_ix = np.empty(nparams, int)
    for i in range(nparams-1, -1, -1):  # reverse loop
        iparam = fvals_shape[i]
        xmin_ix[i] = indx % fvals_shape[i]
        indx = indx // iparam
    # and the corresponding coordinates
    xmin = np.empty(nparams, float)
    for i in range(nparams):
        xmin[i] = grid[i][tuple(xmin_ix)]

    fval_min = fvals_grid[tuple(xmin_ix)]

    if (nparams==1):
        grid = grid[0]
        xmin = xmin[0]

    return xmin, fval_min


def random_minimizer(fun, bounds, N=10, args=(), random_state=None, workers=1):
    """Random search minimizer.

    Arguments
    ---------
    fun: callable
        Objective function to minimize.
    N: int
        No. of random points to sample along the axes.
    bounds: tuple
        Tuple where each element has of the form
        '(low, high)' with the corresponding parameter
        ranges.
    args: tuple, optional
        Additional fixed arguments of the objective
        function.
    random_state:  RandomState instance or None, default=None
        Determines the random number generator used to sample
        points from a uniform distribution along the axes.
    workers: int, default=1
        If > 1, the grid is subdivided into 'workers'
        sections and evaluated in parallel.

    Returns
    -------
    xmin: ndarray
        Coordinates of the point which minimizes
        the objective function.
    fval: float
        Value of the objective function at 'xmin'.
    """
    random_state = check_random_state(random_state)

    nparams = len(bounds)

    rand_pts = np.empty((nparams, N))
    for i in range(nparams):
        rand_pts[i, :] = random_state.uniform(bounds[i][1], bounds[i][0], N)

    grid = np.meshgrid(*rand_pts)
    if (nparams == 1):
        grid = np.squeeze(grid)

    # reshape the grid to get the sets of nparams grid points
    mgrid_shape = np.shape(grid)
    if (nparams > 1):
        grid = np.reshape(grid, (mgrid_shape[0], np.prod(mgrid_shape[1:]))).T

    wrapped_fun = _Fun_Wrapper(fun, args)

    with MapWrapper(pool=workers) as mapper:
        fvals_grid = np.array(list(mapper(wrapped_fun, grid)))
        if (nparams == 1):
            grid = (grid,)
            fvals_grid = np.squeeze(fvals_grid)
        elif (nparams > 1):
            grid = np.reshape(grid.T, mgrid_shape)
            fvals_grid = np.reshape(fvals_grid, mgrid_shape[1:])

    fvals_shape = np.shape(fvals_grid)

    indx = np.argmin(fvals_grid.ravel(), axis=-1)
    # find the index corresponding to the min. value
    xmin_ix = np.empty(nparams, int)
    for i in range(nparams-1, -1, -1):  # reverse loop
        iparam = fvals_shape[i]
        xmin_ix[i] = indx % fvals_shape[i]
        indx = indx // iparam
    # and the corresponding coordinates
    xmin = np.empty(nparams, float)
    for i in range(nparams):
        xmin[i] = grid[i][tuple(xmin_ix)]

    fval_min = fvals_grid[tuple(xmin_ix)]

    if (nparams==1):
        grid = grid[0]
        xmin = xmin[0]

    return xmin, fval_min


def hybrid_minimizer(fun, bounds, N_rand=10, N_brute=10, args=(),
                     random_state=None, workers=1):
    """Hybrid minimizer, starting with a random search to find a
    minimum, and polishing with a grid search around the minimum.

    Arguments
    ---------
    fun: callable
        Objective function to minimize.
    N_rand: int
        No. of random points to sample along the axes for the
        initial search.
    N_brute: int
        No. of grid points along the axes for the polishing search.
    bounds: tuple
        Tuple where each element has of the form
        '(low, high)' with the corresponding parameter
        ranges.
    args: tuple, optional
        Additional fixed arguments of the objective
        function.
    random_state:  RandomState instance or None, default=None
        Determines the random number generator used to sample
        points from a uniform distribution along the axes.
    workers: int, default=1
        If > 1, the grid is subdivided into 'workers'
        sections and evaluated in parallel.

    Returns
    -------
    xmin: ndarray
        Coordinates of the point which minimizes
        the objective function.
    fval: float
        Value of the objective function at 'xmin'.
    """
    # random search
    xmin, fval_min = random_minimizer(fun=fun, N=N_rand,
                                      bounds=bounds,
                                      args=args,
                                      random_state=random_state,
                                      workers=workers)

    if N_brute > 0:
        # grid search centered on xmin
        nparams = len(bounds)
        bounds = np.array(bounds)
        dcell = np.array([(bounds[i][1] - bounds[i][0]) for i in range(nparams)],
                         dtype=float)
        dcell /= (N_rand + 1)
        brute_low = np.max((xmin - dcell / 2.0, bounds[:, 0]), axis=0)
        brute_high = np.min((xmin + dcell / 2.0, bounds[:, 1]), axis=0)
        brute_bounds = list(zip(brute_low, brute_high))
        brute_bounds = [x for x in brute_bounds if \
            within_bounds(x, bounds)]
        xmin, fval_min = brute_minimizer(fun=fun, N=N_brute,
                                         bounds=brute_bounds,
                                         args=args,
                                         workers=workers)
    return xmin, fval_min

def within_bounds(x, bounds):
    x = np.asarray(x)
    return np.all(x >= bounds[:,0]) and np.all(x <= bounds[:, 1])

class _Fun_Wrapper(object):

    def __init__(self, f, args):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        # flatten needed for one dimensional case.
        return self.f(np.asarray(x).flatten(), *self.args)
