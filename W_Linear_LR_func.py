import sys
import numpy
from utils import *
from evaluator import *
from plotting import *
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy.linalg
import sklearn.datasets
import scipy.optimize as opt
from prettytable import PrettyTable


#!tune lambda parameter
def kfold_WEIGHTED_LR_tuning(DTR, LTR, l, PCA_Flag=False, gauss_Flag=False, zscore_Flag=False, pi=0.5):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    LR_labels = []

    #!Kfold approach
    for i in range(k):
        D = []
        L = []
        if i == 0:
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[i + 1:]))
        elif i == k - 1:
            D.append(np.hstack(Dtr[:i]))
            L.append(np.hstack(Ltr[:i]))
        else:
            D.append(np.hstack(Dtr[:i]))
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[:i]))
            L.append(np.hstack(Ltr[i + 1:]))

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = Dtr[i]
        Lte = Ltr[i]

        if zscore_Flag is True:
            D, Dte = znorm(D, Dte)

        if gauss_Flag is True:
            D_training = D
            D = gaussianize_features(D, D)
            Dte = gaussianize_features(D_training, Dte)

        scores = weighted_logistic_reg_score(D, L, Dte, l, pi)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    return np.hstack(scores_append), LR_labels



def weighted_logreg_obj_wrap(DTR, LTR, l, pi=0.5):
    M = DTR.shape[0]
    Z = LTR * 2.0 - 1.0

    def logreg_obj(v):
        w = mcol(v[0:M])
        b = v[-1]
        reg = 0.5 * l * numpy.linalg.norm(w) ** 2
        s = (numpy.dot(w.T, DTR) + b).ravel()
        nt = DTR[:, LTR == 0].shape[1]
        avg_risk_0 = (numpy.logaddexp(0, -s[LTR == 0] * Z[LTR == 0])).sum()
        avg_risk_1 = (numpy.logaddexp(0, -s[LTR == 1] * Z[LTR == 1])).sum()
        return reg + (pi / nt) * avg_risk_1 + (1-pi) / (DTR.shape[1]-nt) * avg_risk_0
    return logreg_obj


def weighted_logistic_reg_score(DTR, LTR, DTE, l, pi=0.5):
    logreg_obj = weighted_logreg_obj_wrap(numpy.array(DTR), LTR, l, pi)
    _v, _J, _d = opt.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]
    STE = numpy.dot(_w.T, DTE) + _b
    return STE


def validate_LR(scores, LR_labels, appendToTitle, l, pi=0.5):
    scores_append = np.hstack(scores)
    scores_tot_05 = compute_min_DCF(scores_append, LR_labels, 0.5, 1, 1)
    scores_tot_01 = compute_min_DCF(scores_append, LR_labels, 0.1, 1, 1)
    scores_tot_09 = compute_min_DCF(scores_append, LR_labels, 0.9, 1, 1)
    
    plot_ROC(scores_append, LR_labels, appendToTitle + 'WEIGHTED_LR, lambda=' + str(l))

    # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(scores_append, LR_labels, appendToTitle + 'WEIGHTED_LR, lambda=' + str(l), 0.4)

    t = PrettyTable(["Type", "π=0.5", "π=0.1", "π=0.9"])
    t.title = appendToTitle
    t.add_row(['WEIGHTED_LR, lambda=' + str(l) + " π_t=" + str(pi), round(scores_tot_05, 3), round(scores_tot_01, 3), round(scores_tot_09, 3)])
    print(t)
