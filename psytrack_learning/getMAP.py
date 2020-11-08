import copy
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags

from .helper.memoize import memoize
from .helper.jacHessCheck import jacHessCheck
from .helper.helperFunctions import (
    DT_X_D, Dv, DTv, DTinv_v,
    sparse_logdet,
    read_input,
    make_invSigma,
    myblk_diags,
)


def getMAP(dat, hyper, weights, W0=None, learning_rule=None, tol=1e-9,
           showOpt=0):
    '''Estimates psychophysical weights with a random walk prior.

    Args:
        dat : dict, all data from a specific subject
        hyper : a dictionary of hyperparameters used to construct the prior
            Must at least include sigma, can also include sigInit, sigDay
        weights : dict, name and count of weights in dat['inputs'] to fit
        W0 : initial parameter estimate, must be of approprite size N*K, 
            defaults to zeros
        learning_rule : fun, the function for computing the learning rule
        tol : float, the tolerance of the weight optimization (higher is less
            accurate, but faster)
        showOpt : {0 : no text, 1 : verbose, 2+ : Hess + deriv check, done
            showOpt-1 times}

    Returns:
        wMode : MAP estimate of the weights, ordered alphabetically as
            specified in `weights`.
        Hess : the Hessian of the log posterior at wMode, used for Laplace appx.
            in evidence max in this case, is a dict of sparse terms needed to
            construct Hess (which is not sparse)
        logEvd : log of the evidence
        llstruct : dictionary containing the components of the log evidence and
            other info
    '''

    # -----
    # Initializations and Sanity Checks
    # -----
    dat = copy.deepcopy(dat)
    
    # Check and count trials
    if 'inputs' not in dat or 'y' not in dat or type(
            dat['inputs']) is not dict:
        raise Exception('getMAP_PBups: insufficient input, missing y')
    N = len(dat['y'])
    
    # Check validity of 'y', must be 1 and 2 (fix automatically if 0 and 1)
    if np.array_equal(np.unique(dat['y']), [0, 1]):
        dat['y'] += 1
    if not np.array_equal(np.unique(dat['y']), [1, 2]):
        raise Exception('getMAP_PBups: y must be parametrized as 1 and 2 only.')

    # Check and count weights
    K = 0
    if type(weights) is not dict:
        raise Exception('weights must be a dict')
    for i in weights.keys():
        if type(weights[i]) is not int or weights[i] < 0:
            raise Exception('weight values must be non-negative ints')
        K += weights[i]

    # Initialize weights to particular values (default 0)
    w_N = N
    if W0 is not None:
        if type(W0) is not np.ndarray:
            raise Exception('W0 must be an array')

        if W0.shape == (w_N * K,):
            wInit = W0.copy()
        elif W0.shape == (w_N, K):
            wInit = W0.flatten()
        else:
            raise Exception('W0 must be shape (w_N*K,) or (w_N,K), not ' +
                            str(W0.shape))
    else:
        wInit = np.zeros(w_N * K)

    # Do sanity checks on hyperparameters
    if 'sigma' not in hyper:
        raise Exception('WARNING: sigma not specified in hyper dict')

    # Get index of start of each day
    if ('dayLength' not in dat) and (
        ('sigDay' in hyper and hyper['sigDay'] is not None)):
        print('WARNING: sigDay has no effect, dayLength not supplied in dat')
        dat['dayLength'] = np.array([], dtype=int)

    # -----
    # MAP estimate
    # -----

    # Prepare minimization of loss function, Memoize to preserve Jac+Hess info
    lossfun = memoize(negLogPost)
    my_args = (dat, hyper, weights, learning_rule)

    if showOpt:
        opts = {'disp': True}
        if int(showOpt) > 2:
            opts['maxiter'] = 5
        callback = print
    else:
        opts = {'disp': False}
        callback = None

    # Actual optimization call
    # Uses 'hessp' to pass a function that calculates product of Hessian
    #    with arbitrary vector
    if showOpt:
        print('Obtaining MAP estimate...')
    result = minimize(
        lossfun,  # function which returns value to minimize
        wInit,    # input to function to optimize over
        jac=lossfun.jacobian,
        hessp=lossfun.hessian_prod,
        method='Newton-CG',
        tol=tol,
        args=my_args,  # OTHER inputs to the function which remain fixed
        options=opts,
        callback=callback,
    )

    # Recover the results of the optimization
    wMode = result.x
    # The Hessian at wMode
    Hess = lossfun.hessian(wMode, *my_args)

    # Print message if optimizer does not converge (usually still pretty good)
    if showOpt and not result.success:
        print('WARNING — MAP estimate: minimize() did not converge\n',
              result.message)
        print('NOTE: this is ususally irrelevant as the optimizer still finds '
              'a good solution. If you are concerned, run a check of the '
              'Hessian by setting showOpt >= 2')

    # Run DerivCheck & HessCheck at eMode (will run ShowOpt-1 distinct times)
    if showOpt >= 2:
        print('** Jacobian and Hessian Check **')
        for check in range(showOpt - 1):
            print('\nCheck', check + 1, ':')
            jacHessCheck(lossfun, wMode, *my_args)
            print('')

    # -----
    # Evidence (Marginal likelihood)
    # -----

    # Prior and likelihood at eMode, also recovering the associated wMode
    if showOpt:
        print('Calculating evd, first prior and likelihood at wMode...')
    pT, lT = getPosteriorTerms(wMode, *my_args)

    # Posterior term (with Laplace approx), calculating sparse log determinant
    if showOpt:
        print('Now the posterior with Laplace approx...')
    logterm_post = (1 / 2) * sparse_logdet(Hess)

    # Compute Log evd and construct dict of likelihood, prior,
    #   and posterior terms
    logEvd = lT['logli'] + pT['logprior'] - logterm_post
    if showOpt:
        print('Evidence:', logEvd)

    # Package up important terms to return
    llstruct = {'lT': lT, 'pT': pT, 'post': logterm_post}

    # wMode = wMode.reshape((K, w_N), order="C")
    return wMode, Hess, logEvd, llstruct


def negLogPost(*args):
    '''Returns negative log posterior (and its first and second derivative)
    Intermediary function to allow for getPosteriorTerms to be optimized

    Args:
        same as getPosteriorTerms()

    Returns:
        negL : negative log-likelihood of the posterior
        dL : 1st derivative of the negative log-likelihood
        ddL : 2nd derivative of the negative log-likelihood,
            kept as a dict of sparse terms!
    '''

    # Get prior and likelihood terms
    priorTerms, liTerms = getPosteriorTerms(*args)  # pylint: disable=no-value-for-parameter

    # Negative log posterior
    negL = - priorTerms['logprior']   - liTerms['logli']
    dL   = - priorTerms['dlogprior']  - liTerms['dlogli']
    ddL  = - priorTerms['ddlogprior'] - liTerms['ddlogli']

    return negL, dL, ddL


def getPosteriorTerms(W_flat, dat, hyper, weights, learning_rule=None):
    '''Given a sequence of parameters formatted as an N*K matrix, calculates
    random-walk log priors & likelihoods and their derivatives

    Args:
        W_flat : array, the N*K weight parameters, flattened to a single
        vector
        ** all other args are same as in getMAP **

    Returns:
        priorTerms : dict, the log-prior as well as 1st + 2nd derivatives
        liTerms : dict, the log-likelihood as well as 1st + 2nd derivatives
    '''

    # ---
    # Initialization
    # ---

    # If function is called directly instead of through getMAP,
    #       fill in dummy values
    if 'dayLength' not in dat:
        dat['dayLength'] = np.array([], dtype=int)

    # Unpack input into g
    g = read_input(dat, weights)
    N, K = g.shape

    # the first trial index of each new day
    days = np.cumsum(dat['dayLength'], dtype=int)[:-1]

    # Check shape of epsilon, with
    #   w_N (effective # of trials) * K (# of weights) elements
    if W_flat.shape != (N * K,):
        print(W_flat.shape, N, K)
        raise Exception('parameter dimension mismatch (#trials * #weights)')
    W = np.reshape(W_flat, (K, N), order='C')
    
        
    # ---
    # Construct random-walk prior, calculate priorTerms
    # ---
        
    # Construct random walk covariance matrix Sigma^-1, use sparsity for speed
    invSigma_diag = make_invSigma(hyper, days, None, N, K)
    logdet_invSigma = np.sum(np.log(invSigma_diag))
    E_flat = Dv(W_flat, K) 
    
    # Calculate the the log-prior, 1st, & 2nd derivatives
    if learning_rule is None:
        logprior = (1 / 2) * (logdet_invSigma - (E_flat**2 * invSigma_diag).sum())
        dlogprior = -DTv(invSigma_diag * E_flat, K)
        ddlogprior = -DT_X_D(diags(invSigma_diag), K)
        learning_terms = None
    # Account for policy update "drift" (learning)
    else:
        v, dvdw, ddvdwdw = learning_rule(W=W, X=g, dat=dat, hyper=hyper)

        E_flat = E_flat - v.flatten(order="F")
        logprior = (1 / 2) * (logdet_invSigma - (E_flat**2 * invSigma_diag).sum())
        
        offdiag = np.ones((K, N)) 
        offdiag[:, -1] = 0
        offdiag = offdiag.flatten()[:-1]
        aux = myblk_diags(dvdw, offset=-1)
        aux.setdiag(aux.diagonal() - 1)
        aux.setdiag(aux.diagonal(k=-1) + offdiag, k=-1)
        aux = aux.tocsr()
                
        aux2 = E_flat * invSigma_diag
        dlogprior = aux2 @ aux
        
        ddlogprior = -aux.T.multiply(invSigma_diag) @ aux
        diag_vals = 0
        for k in np.arange(K):
            diag_vals += aux2 @ myblk_diags(ddvdwdw[:,k], offset=-1)
        ddlogprior.setdiag(ddlogprior.diagonal() + diag_vals)
                    
        learning_terms = {"v": v, "E_flat": E_flat}

    priorTerms = {
        'logprior': logprior,
        'dlogprior': dlogprior,
        'ddlogprior': ddlogprior,
        'learning_terms' : learning_terms
    }

    # ---
    # Construct likelihood, calculate liTerms
    # ---

    # Calculate probability of Right on each trial
    y = dat['y'] - 1
    gw = np.sum(g * W.T, axis=1)
    pR = 1 / (1 + np.exp(-gw))

    # Preliminary calculations for 1st and 2nd derivatives
    dlliList = g * (y - pR)[:, None]
    HlliList = (pR**2 - pR)[:, None, None] * (g[:, :, None] @ g[:, None, :])

    # Calculate the log-likelihood and 1st & 2nd derivatives
    logli = np.sum(y * gw - np.logaddexp(0, gw))
    dlogli = dlliList.flatten('F')
    ddlogli = myblk_diags(HlliList)

    liTerms = {'logli': logli, 'dlogli': dlogli, 'ddlogli': ddlogli}

    return priorTerms, liTerms
