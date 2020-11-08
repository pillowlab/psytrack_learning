import numpy as np
from scipy.special import expit


def reward_max(W, X, y, r, answer, i, base=None):
    pR = expit(X[i-1] @ W[i-1])
    return pR * (1-pR) * X[i-1] * (-1)**(answer[i-1]+1) / (1/2)


def predict_max(W, X, y, r, answer, i, base=None):
    pCorrect = np.abs((1-answer[i-1]) - expit(X[i-1] @ W[i-1]))
    return (1-pCorrect) * X[i-1] * (-1)**(answer[i-1]+1) / 1


def reinforce(W, X, y, r, answer, i, base=None):
    pChoice = np.abs((1-y[i-1]) - expit(X[i-1] @ W[i-1]))
    return (1-pChoice) * X[i-1] * (-1)**(y[i-1]+1) * r[i-1]


def reinforce_base(W, X, y, r, answer, i, base):
    pChoice = np.abs((1-y[i-1]) - expit(X[i-1] @ W[i-1]))
    return (1-pChoice) * X[i-1] * (-1)**(y[i-1]+1) * (r[i-1] - base)


def simulate_learning(X, answer, sigma, alpha, learning_rule, 
                      base=0, sigma0=0, W0=0, seed=None):
    '''Simulates weights, choices, and rewards for a given task and learning rule.
    
    Args:
        X: (N x K) array of inputs, best taken from real inputs given to example animals
        answer: array of N correct choices for the given X
        sigma: either a single value or an array of length K, the smoothness of each weight
        alpha: either a single value or an array of length K, the learning rate of each weight
        learning_rule: a function which returns the update to the weights after each trial
        sigma0: the initial sigma of each weight, controlling distance of initialization from 0
    
    Returns:
        W: (N x K) array of weights
        y: array of N choices {0, 1}
        r: array of N rewards {0, 1}
    '''
    
    N, K = X.shape

    # Can calculate the noise added to each weight on each trial in advance
    np.random.seed(seed)
    noise = np.random.normal(scale=sigma, size=(N, K))
    noise[0] = np.random.normal(scale=sigma0, size=K)

    # Inputs
    W = np.zeros((N,K))  # weights
    y = np.zeros(N)      # choice {0,1}
    r = np.zeros(N)      # reward {0,1}

    # Initialize weights, choice, and reward on first trial
    W[0] = noise[0] + W0
    y[0] = (np.random.rand() < expit(X[0] @ W[0])).astype(int)
    r[0] = (y[0]==answer[0]).astype(int)

    # Iterate through remaining N-1 trials
    for i in range(1,N):

        # Calculate the learning update from the last trial
        learning_update = learning_rule(W, X, y, r, answer, i, base)

        # Update the weights
        W[i] = W[i-1] + noise[i] + alpha*learning_update

        # Calculate choice on current trial
        y[i] = (np.random.rand() < expit(X[i] @ W[i])).astype(int)

        # Calculate reward
        r[i] = (y[i]==answer[i]).astype(int)

    return W.T, y, r, noise