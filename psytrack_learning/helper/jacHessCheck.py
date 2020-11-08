import numpy as np
from scipy.sparse.linalg import spsolve, inv


def jacHessCheck(fun, x0, *args, **kwargs):
    '''
    Checks the accuracy of the analytic Jacobian and Hessian
    of a function that is part of the Memoize class by calculating
    finite differences in a tiny random direction.

    Args:
        fun : a function returning [result, jac, hess] that has
            been passed through a Memoize class
        x0 : the point from which to evaluate the function

    Returns:
        Nothing, simply prints out the analytic and finite
        differencing results for both the jac and hess
    '''

    fun(x0, *args, **kwargs)
    JJ = fun.jacobian(x0, *args, **kwargs)
    HH = fun.hessian(x0, *args, **kwargs)

    tol = 1e-6
    randjump = np.random.rand(len(x0)) * tol

    f1 = fun(x0 - randjump / 2, *args, **kwargs)
    JJ1 = fun.jacobian(x0 - randjump / 2, *args, **kwargs)

    f2 = fun(x0 + randjump / 2, *args, **kwargs)
    JJ2 = fun.jacobian(x0 + randjump / 2, *args, **kwargs)

    print('Analytic Jac:', np.dot(randjump, JJ))
    print('Finite Jac:  ', f2 - f1)

    if type(HH) is dict:
        print('Analytic Hess:',
              np.sum(fun.hessian_prod(x0, randjump, *args, **kwargs)))
    else:
        print('Analytic Hess:', np.sum(HH @ randjump))
    print('Finite Hess:  ', np.sum(JJ2 - JJ1))
    

def compHess(fun, x0, dx, kwargs):
    '''Numerically computes the Hessian of a function fun around point x0.
    
    Expects fun to have sytax:  y = fun(x, varargin)

    Args:
        fun: @(x) function handle of a real valued function that takes column vector
        x0: (n x 1) point at which Hessian and gradient are estimated
        dx: (1) or (n x 1) step size for finite difference
        kwargs: extra arguments are passed to the fun

    Returns:
        H: Hessian estimate
        g: gradient estiamte
    '''

    n = len(x0)
    H = np.zeros((n, n))
    g = np.zeros(n)
    f0 = fun(x0, **kwargs)

    vdx = dx*np.ones(n)
    A = np.diag(vdx/2.0)

    for j in range(n):  # compute diagonal terms
        # central differences
        f1 = fun(x0 + 2*A[:, j], **kwargs)
        f2 = fun(x0 - 2*A[:, j], **kwargs)
        H[j,j] = f1 + f2 - 2*f0
        g[j] = (f1 - f2)/2

    for j in range(n-1):  # compute cross terms
        for i in range(j+1, n):
            # central differences
            f11 = fun(x0 + A[:, j] + A[:, i], **kwargs)
            f22 = fun(x0 - A[:, j] - A[:, i], **kwargs)
            f12 = fun(x0 + A[:, j] - A[:, i], **kwargs)
            f21 = fun(x0 - A[:, j] + A[:, i], **kwargs)
            H[j, i] = f11 + f22 - f12 - f21
            H[i, j] = H[j, i]

    H = H / dx / dx
    g = g / dx
    
    return H, g


def compHess_nolog(fun, x0, dx, kwargs):
    '''Numerically computes the Hessian of a function fun around point x0.
    
    Computation is done in non-log space for all hyperparameters, though
    some are optimized in log2 space.

    Args:
        fun: @(x) function handle of a real valued function that takes column vector
        x0: (n x 1) point at which Hessian and gradient are estimated
        dx: (1) or (n x 1) step size for finite difference
        kwargs: extra arguments are passed to the fun

    Returns:
        H: Hessian estimate
        g: gradient estiamte
    '''

    n = len(x0)
    H = np.zeros((n, n))
    g = np.zeros(n)
    
    optList = kwargs['keywords']['optList']
    K = kwargs['keywords']['K']
    hyper = kwargs['keywords']['hyper']
    
    def unlog(nolog_x0, optList, K):
        x0 = []
        count = 0
        for h in optList:
            if h in ["adder"]: # no 2**
                if np.isscalar(hyper[h]):
                    x0 += [nolog_x0[count]]
                    count += 1
                else:
                    x0 += nolog_x0[count:count + K].tolist()
                    count += K
            elif h in ["sigma", "alpha", "sigInit"]:
                if np.isscalar(hyper[h]):
                    x0 += [np.log2(nolog_x0[count])]
                    count += 1
                else:
                    x0 += np.log2(nolog_x0[count:count + K]).tolist()
                    count += K
        return np.array(x0)
    
    nolog_x0 = []
    count = 0
    for h in optList:
        if h in ["adder"]: # no 2**
            if np.isscalar(hyper[h]):
                nolog_x0 += [x0[count]]
                count += 1
            else:
                nolog_x0 += x0[count:count + K].tolist()
                count += K
        elif h in ["sigma", "alpha", "sigInit"]:
            if np.isscalar(hyper[h]):
                nolog_x0 += [2**x0[count]]
                count += 1
            else:
                nolog_x0 += (2**x0[count:count + K]).tolist()
                count += K
    
    nolog_x0 = np.array(nolog_x0)
    
    f0 = fun(x0, **kwargs)

    vdx = dx*nolog_x0
    A = np.diag(vdx/2.0)

    for j in range(n):  # compute diagonal terms
        # central differences
        f1 = fun(unlog(nolog_x0 + 2*A[:, j], optList, K), **kwargs)
        f2 = fun(unlog(nolog_x0 - 2*A[:, j], optList, K), **kwargs)
        H[j,j] = f1 + f2 - 2*f0
        g[j] = (f1 - f2)/2

    for j in range(n-1):  # compute cross terms
        for i in range(j+1, n):
            # central differences
            f11 = fun(unlog(nolog_x0 + A[:, j] + A[:, i], optList, K), **kwargs)
            f22 = fun(unlog(nolog_x0 - A[:, j] - A[:, i], optList, K), **kwargs)
            f12 = fun(unlog(nolog_x0 + A[:, j] - A[:, i], optList, K), **kwargs)
            f21 = fun(unlog(nolog_x0 - A[:, j] + A[:, i], optList, K), **kwargs)
            H[j, i] = f11 + f22 - f12 - f21
            H[i, j] = H[j, i]

    H = H / np.outer(vdx, vdx)
    g = g / vdx
    
    return H, g
