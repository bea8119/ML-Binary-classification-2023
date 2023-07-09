import sys
import numpy as np
from Quadratic_LR_func import *

from utils import *
from prettytable import PrettyTable
from plotting import *
from evaluator import *

#K fold approach
def kfold_QUAD_LR(DTR, LTR, l, pi, appendToTitle, PCA_Flag=True, gauss_Flag=False, zscore_Flag=False):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    PCA_LR_scores_append = []
    PCA2_LR_scores_append = []
    LR_labels = []

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

        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
            return xxT

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = Dtr[i]
        Lte = Ltr[i]

        if zscore_Flag is True:
            D, Dte = znorm(D, Dte)

        if gauss_Flag is True:
            Dte = gaussianize_features(D, Dte)
            D = gaussianize_features(D, D)

        expanded_DTR = numpy.apply_along_axis(vecxxT, 0, D)
        expanded_DTE = numpy.apply_along_axis(vecxxT, 0, Dte)
        phi = numpy.vstack([expanded_DTR, D])

        phi_DTE = numpy.vstack([expanded_DTE, Dte])

        scores = quad_logistic_reg_score(phi, L, phi_DTE, l, pi)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

        if PCA_Flag is True:
            # PCA m=11
            P = PCA(D, L, m=11)
            DTR_PCA = numpy.dot(P.T, D)
            DTE_PCA = numpy.dot(P.T, Dte)

            PCA_LR_scores = quad_logistic_reg_score(DTR_PCA, L, DTE_PCA, l, pi)
            PCA_LR_scores_append.append(PCA_LR_scores)

            # PCA m=10
            P = PCA(D, L, m=10)
            DTR_PCA = numpy.dot(P.T, D)
            DTE_PCA = numpy.dot(P.T, Dte)

            PCA2_LR_scores = quad_logistic_reg_score(DTR_PCA, L, DTE_PCA, l)
            PCA2_LR_scores_append.append(PCA2_LR_scores)

    validate_LR(scores_append, LR_labels, appendToTitle, l, pi)

    if PCA_Flag is True:
        validate_LR(PCA_LR_scores_append, LR_labels, appendToTitle + 'PCA_m11_', l, pi)

        validate_LR(PCA2_LR_scores_append, LR_labels, appendToTitle + 'PCA_m10_', l, pi)


def validation_quad_LR(DTR, LTR, L, appendToTitle, PCA_Flag=True, gauss_Flag=False, zscore_Flag=False):
    for l in L:
        kfold_QUAD_LR(DTR, LTR, l, 0.5, appendToTitle, PCA_Flag, gauss_Flag, zscore_Flag)
        kfold_QUAD_LR(DTR, LTR, l, 0.1, appendToTitle, PCA_Flag, gauss_Flag, zscore_Flag)
        kfold_QUAD_LR(DTR, LTR, l, 0.9, appendToTitle, PCA_Flag, gauss_Flag, zscore_Flag)

    x = numpy.logspace(-5, 1, 20)
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])
    for xi in x:
        scores, labels = kfold_QUAD_LR_tuning(DTR, LTR, xi, PCA_Flag, gauss_Flag, zscore_Flag)
        y_05 = numpy.hstack((y_05, bayes_error_plot_compare(0.5, scores, labels)))
        y_09 = numpy.hstack((y_09, bayes_error_plot_compare(0.9, scores, labels)))
        y_01 = numpy.hstack((y_01, bayes_error_plot_compare(0.1, scores, labels)))

    y = numpy.hstack((y, y_05))
    y = numpy.vstack((y, y_09))
    y = numpy.vstack((y, y_01))

    plot_DCF(x, y, 'lambda', appendToTitle + 'QUAD_LR_minDCF_comparison')


if __name__ == "__main__":
    
    #load and randomize TRAINING set
    DTR, LTR = load("dataset/Train.txt")
    DTR, LTR = randomize(DTR, LTR)
    
    #load and randomize TEST set
    DTE, LTE = load("dataset/Test.txt")
    DTE, LTE = randomize(DTE, LTE)
    
    
    print("############    Quadratic Logistic Regression    ##############")
    L = [1e-5]  #lambda regularization term
    validation_quad_LR(DTR, LTR, L, 'QUAD_', PCA_Flag=True, gauss_Flag=False, zscore_Flag=False)        #RAW features
    validation_quad_LR(DTR, LTR, L, 'GAUSSIANIZED_', PCA_Flag=True, gauss_Flag=True, zscore_Flag=False) #Gaussianized features
    validation_quad_LR(DTR, LTR, L, 'ZNORM_', PCA_Flag=True, gauss_Flag=False, zscore_Flag=True)        #Z-normed features

