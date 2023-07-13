from utils import *
from DCF import DCF_unnormalized_normalized_min_binary
import numpy as np
import scipy.optimize


def calibrate_scores(scores_D, L, eff_prior, w=None, b=None):
    
    scores_D = mrow(scores_D)
    if w is None and b is None:

        scores, w, b = scores(scores_D, L, 0, eff_prior, scores_D, calibrate=True)
        calibrated_scores = scores - np.log(eff_prior / (1 - eff_prior))
        return calibrated_scores, w, b
    else:
        calibrated_scores = np.dot(w.T, scores_D) + b
        return calibrated_scores.ravel()


def scores(DTR, LTR, l, priorT, DTE, calibrate=False):
    x0 = np.zeros(DTR.shape[0] + 1)

    objbin=logreg_obj_binary(DTR, LTR, l, priorT)

    (v, J, d) = scipy.optimize.fmin_l_bfgs_b(
        objbin,
        x0,
        approx_grad=True,
        factr=1.0,
        # maxiter=100,
    )
    w = mcol(v[0:-1])
    b = v[-1]
    p_lprs = np.dot(w.T, DTE) + b  # Posterior log-probability ratio
    if calibrate:
        return p_lprs.ravel(), w, b
    return p_lprs.ravel()
    

def logreg_obj_binary(DTR, LTR, l, priorT, v):
    w, b = mcol(v[0:-1]), v[-1]
    s0 = np.dot(w.T, DTR[:, LTR == 0]) + b
    s1 = np.dot(w.T, DTR[:, LTR == 1]) + b
    # Cross-entropy
    # directly z=1 since it is the true samples
    mean_term1 = np.logaddexp(0, -s1).mean()
    mean_term0 = np.logaddexp(0, s0).mean()
    crossEntropy = priorT * mean_term1 + \
        (1 - priorT) * mean_term0
    return 0.5 * l * np.linalg.norm(w)**2 + crossEntropy