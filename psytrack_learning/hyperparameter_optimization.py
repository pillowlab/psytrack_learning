import numpy as np

from .getMAP import getMAP
from .helper.helperFunctions import update_hyper

def evd_lossfun(vals, keywords):

    # Update the hyper dict with the current guesses
    hyper = update_hyper(vals, keywords['optList'], keywords['hyper'], keywords['K'])
    lr = keywords['learning_rule']
    # lr = lr if lr is None else globals()[lr]
    
    # Recover the weights and evidence for the current hyper guess
    wMode, _, logEvd, _ = getMAP(
        keywords['dat'],
        hyper,
        keywords['weights'],
        W0=keywords["wMode"],
        learning_rule=lr,
        tol=keywords["tol"],
        showOpt=0)

    # When optimizing hypers, update the initial guess of weights for the next iteration
    if keywords["update_w"]:
        keywords["wMode"] = wMode
        if "iter" in keywords:
            keywords["iter"] += 1
        else:
            keywords["iter"] = 1
    else:
        keywords["iter"] = 0
    
    # Print fitting info if desired
    if "showOpt" in keywords and keywords["showOpt"]:
        print("   ", np.round(vals,3), keywords["iter"], np.round(-logEvd,3),
              "")
        
    return -logEvd