import numpy as np


def RewardMax(W, X, dat, hyper, *args):
    '''
    Compute trial-specific policy gradient (and its gradient w.r.t. weight)
    at a particluar time point t according to the RewardMax learning rule.
    
    Args:
        W: the weights
        X: the inputs
        dat: dict of data
        alpha: float or array of learning rates for each weight

    Returns:
        mypg : the policy gradient
        dpg : 1st derivative of the policy gradient w.r.t. w
        ddpg : 2nd derivative of the policy gradient w.r.t. w
    '''
    _, K = X.shape
    if np.isscalar(hyper['alpha']):
        alpha = np.array([hyper['alpha']] * K)
    else:
        alpha = np.array(hyper['alpha'])
        
    answer = dat['answer']
    if 2 in answer:
        answer = answer - 1
    
    pR = 1 / (1 + np.exp(-np.sum(W.T * X, axis=1)))  # P(Right)
    f = (-1)**(answer+1)  # -1 if correct answer is left, +1 if right
    normalizer = 0.5  # the max value of pR*(1-pR)

    ### Policy gradient
    x = X.T
    coeff1 = f * pR * (1 - pR) / normalizer
    v = coeff1[:,None] * x.T * alpha

    ### First derivative
    xx = X.T[:, None] * X.T[None, :]
    coeff2 = f * pR * (1 - pR) * (1 - 2*pR) / normalizer
    dvdw = coeff2[:,None,None] * xx.T * alpha[:,None]

    ### Second derivative
    xxx = X.T[None, None, :] * X.T[None, :, None] * X.T[:, None, None]
    coeff3 = f * pR * (1 - pR) * (1 - 6*pR + 6*pR**2) / normalizer
    ddvdwdw = coeff3[:,None,None,None] * xxx.T * alpha[:,None]
        
    # Return policy gradient, it's 1st derivative, and its 2nd derivative
    v = np.vstack((np.zeros(K), v[:-1]))
    dvdw = np.vstack((np.zeros((1,K,K)), dvdw[:-1]))
    ddvdwdw = np.vstack((np.zeros((1,K,K,K)), ddvdwdw[:-1]))
    return v, dvdw, ddvdwdw


def PredictMax(W, X, dat, hyper, *args):
    '''
    Compute trial-specific policy gradient (and its gradient w.r.t. weight)
    at a particluar time point t according to the PredictMax learning rule.
    
    Args:
        W: the weights
        X: the inputs
        dat: dict of data
        alpha: float or array of learning rates for each weight

    Returns:
        mypg : the policy gradient
        dpg : 1st derivative of the policy gradient w.r.t. w
        ddpg : 2nd derivative of the policy gradient w.r.t. w
    '''
    _, K = X.shape
    if np.isscalar(hyper['alpha']):
        alpha = np.array([hyper['alpha']] * K)
    else:
        alpha = np.array(hyper['alpha'])

    answer = dat['answer']
    if 2 in answer:
        answer = answer - 1
        
    pR = 1 / (1 + np.exp(-np.sum(W.T * X, axis=1)))  # P(Right)
    pC = np.abs((1-answer) - pR)  # P(Correct Answer)
    f = (-1)**(answer+1)  # -1 if correct answer is left, +1 if right
    normalizer = 1.0  # the max value of (1-pC)

    ### Policy gradient
    x = X.T
    coeff1 = f * (1 - pC) / normalizer
    v = coeff1[:,None] * x.T * alpha

    ### First derivative
    xx = X.T[:, None] * X.T[None, :]
    coeff2 = -pC * (1 - pC) / normalizer
    dvdw = coeff2[:,None,None] * xx.T * alpha[:,None]

    ### Second derivative
    xxx = X.T[None, None, :] * X.T[None, :, None] * X.T[:, None, None]
    coeff3 = f * -pC * (1 - pC) * (1 - 2*pC) / normalizer
    ddvdwdw = coeff3[:,None,None,None] * xxx.T * alpha[:,None]
        
    # Return policy gradient, it's 1st derivative, and its 2nd derivative
    v = np.vstack((np.zeros(K), v[:-1]))
    dvdw = np.vstack((np.zeros((1,K,K)), dvdw[:-1]))
    ddvdwdw = np.vstack((np.zeros((1,K,K,K)), ddvdwdw[:-1]))
    return v, dvdw, ddvdwdw


def REINFORCE(W, X, dat, hyper, *args):
    '''
    Compute trial-specific policy gradient (and its gradient w.r.t. weight)
    at a particluar time point t according to the L2Min learning rule.
    
    Args:
        W: the weights
        X: the inputs
        dat: dict of data
        alpha: float or array of learning rates for each weight

    Returns:
        mypg : the policy gradient
        dpg : 1st derivative of the policy gradient w.r.t. w
        ddpg : 2nd derivative of the policy gradient w.r.t. w
    '''
    _, K = X.shape
    if np.isscalar(hyper['alpha']):
        alpha = np.array([hyper['alpha']] * K)
    else:
        alpha = np.array(hyper['alpha'])

    y = dat['y']
    if 2 in y:
        y = y - 1
        
    pR = 1 / (1 + np.exp(-np.sum(W.T * X, axis=1)))  # P(Right)
    py = np.abs((1-y) - pR)  # P(Choice Made)
    f = (-1)**(y+1)  # -1 if choose left, +1 if right
    r = dat['correct'] # 0 if incorrect answer, +1 if correct
    normalizer = 1  # the max value of (1 - py)

    ### Policy gradient
    x = X.T
    coeff1 = f * (1 - py) * r / normalizer
    v = coeff1[:,None] * x.T * alpha

    ### First derivative
    xx = X.T[:, None] * X.T[None, :]
    coeff2 = -py * (1 - py) * r / normalizer
    dvdw = coeff2[:,None,None] * xx.T * alpha[:,None]

    ### Second derivative
    xxx = X.T[None, None, :] * X.T[None, :, None] * X.T[:, None, None]
    coeff3 = f * -py * (1 - py) * (1 - 2*py) * r / normalizer
    ddvdwdw = coeff3[:,None,None,None] * xxx.T * alpha[:,None]
        
    # Return policy gradient, it's 1st derivative, and its 2nd derivative
    v = np.vstack((np.zeros(K), v[:-1]))
    dvdw = np.vstack((np.zeros((1,K,K)), dvdw[:-1]))
    ddvdwdw = np.vstack((np.zeros((1,K,K,K)), ddvdwdw[:-1]))
    return v, dvdw, ddvdwdw


def REINFORCE_base(W, X, dat, hyper, *args):
    '''
    Compute trial-specific policy gradient (and its gradient w.r.t. weight)
    at a particluar time point t according to the L2Min learning rule.
    
    Args:
        W: the weights
        X: the inputs
        dat: dict of data
        alpha: float or array of learning rates for each weight

    Returns:
        mypg : the policy gradient
        dpg : 1st derivative of the policy gradient w.r.t. w
        ddpg : 2nd derivative of the policy gradient w.r.t. w
    '''
    N, K = X.shape
    if np.isscalar(hyper['alpha']):
        alpha = np.array([hyper['alpha']] * K)
    else:
        alpha = np.array(hyper['alpha'])


    if np.isscalar(hyper['adder']):
        adder = np.array([hyper['adder']] * K)
    else:
        adder = np.array(hyper['adder'])
        
    y = dat['y']
    if 2 in y:
        y = y - 1
        
    pR = 1 / (1 + np.exp(-np.sum(W.T * X, axis=1)))  # P(Right)
    py = np.abs((1-y) - pR)  # P(Choice Made)
    f = (-1)**(y+1)  # -1 if choose left, +1 if right
    r = dat['correct'] # 0 if incorrect answer, +1 if correct
    normalizer = 1  # the max value of (1 - py)
    
    # Alternate parametrization of R+B (more accurate)
    r = np.ones((N, K)) * r[:, None] * alpha - adder
    
    # Standard parametrization of R+B
    # r = (np.ones((N, K)) * r[:, None] - adder) * alpha


    ### Policy gradient
    x = X.T
    coeff1 = f * (1 - py) / normalizer
    v = coeff1[:,None] * x.T * r
    
    ### First derivative
    xx = x[:, None] * x[None, :]
    coeff2 = -py * (1 - py) / normalizer
    dvdw = coeff2[:,None,None] * xx.T * r[:, :, None]

    ### Second derivative
    xxx = x[None, None, :] * x[None, :, None] * x[:, None, None]
    coeff3 = f * -py * (1 - py) * (1 - 2*py) / normalizer
    ddvdwdw = coeff3[:,None,None,None] * xxx.T * r[:, None, :, None]
        
    # Return policy gradient, it's 1st derivative, and its 2nd derivative
    v = np.vstack((np.zeros(K), v[:-1]))
    dvdw = np.vstack((np.zeros((1,K,K)), dvdw[:-1]))
    ddvdwdw = np.vstack((np.zeros((1,K,K,K)), ddvdwdw[:-1]))
    return v, dvdw, ddvdwdw