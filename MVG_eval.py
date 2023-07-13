
import sys
import numpy as np
from utils import *
from MVG_func import *
from evaluator import *
from prettytable import PrettyTable


def compute_MVG_score(Dte, D, L, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels):
    _, _, llrs = MVG(Dte, D, L)
    _, _, llrsn = naive_MVG(Dte, D, L)
    _, _, llrst = tied_cov_GC(Dte, D, L)
    _, _, llrsnt = tied_cov_naive_GC(Dte, D, L)

    MVG_res.append(llrs)
    MVG_naive.append(llrsn)
    MVG_t.append(llrst)
    MVG_nt.append(llrsnt)
    # MVG_labels.append(Lte)
    # MVG_labels = np.append(MVG_labels, Lte, axis=0)
    # MVG_labels = np.hstack(MVG_labels)
    return MVG_res, MVG_naive, MVG_t, MVG_nt


def evaluation(title, pi, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels, appendToTitle):
    MVG_res = np.hstack(MVG_res)
    MVG_naive = np.hstack(MVG_naive)
    MVG_t = np.hstack(MVG_t)
    MVG_nt = np.hstack(MVG_nt)

    llrs_tot = compute_min_DCF(MVG_res, MVG_labels, pi, 1, 1)
    llrsn_tot = compute_min_DCF(MVG_naive, MVG_labels, pi, 1, 1)
    llrst_tot = compute_min_DCF(MVG_t, MVG_labels, pi, 1, 1)
    llrsnt_tot = compute_min_DCF(MVG_nt, MVG_labels, pi, 1, 1)

    # plot_ROC(MVG_res, MVG_labels, appendToTitle + 'MVG')
    # plot_ROC(MVG_naive, MVG_labels, appendToTitle + 'MVG + Naive')
    # plot_ROC(MVG_t, MVG_labels, appendToTitle + 'MVG + Tied')
    # plot_ROC(MVG_nt, MVG_labels, appendToTitle + 'MVG + Naive + Tied')

    # # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(MVG_res, MVG_labels, appendToTitle + 'MVG', 0.4)
    # bayes_error_min_act_plot(MVG_naive, MVG_labels, appendToTitle + 'MVG + Naive', 1)
    # bayes_error_min_act_plot(MVG_t, MVG_labels, appendToTitle + 'MVG + Tied', 0.4)
    # bayes_error_min_act_plot(MVG_nt, MVG_labels, appendToTitle + 'MVG + Naive + Tied', 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = title
    t.add_row(["MVG", round(llrs_tot, 3)])
    t.add_row(["MVG naive", round(llrsn_tot, 3)])
    t.add_row(["MVG tied", round(llrst_tot, 3)])
    t.add_row(["MVG naive + tied", round(llrsnt_tot, 3)])
    print(t)
    
    
    

def evaluation_MVG(DTR, LTR, DTE, LTE, appendToTitle, PCA_Flag=True, Gauss_flag=False, zscore=False):

    MVG_res = []
    MVG_naive = []
    MVG_t = []
    MVG_nt = []
    MVG_labels = []

    PCA_mvg = []
    PCA_mvg_naive = []
    PCA_mvg_t = []
    PCA_mvg_nt = []

    PCA2_mvg = []
    PCA2_mvg_naive = []
    PCA2_mvg_t = []
    PCA2_mvg_nt = []

    if (zscore):
            DTR, DTE = znorm(DTR, DTE)

    if (Gauss_flag):
        D_training = DTR
        DTR = gaussianize_features(DTR, DTR)
        DTE = gaussianize_features(D_training, DTE)


    # RAW DATA
    MVG_labels = np.append(MVG_labels, LTE, axis=0)
    MVG_labels = np.hstack(MVG_labels)

    MVG_res, MVG_naive, MVG_t, MVG_nt = compute_MVG_score(
        DTE,
        DTR,
        LTR,
        MVG_res,
        MVG_naive,
        MVG_t,
        MVG_nt,
        MVG_labels)

    if PCA_Flag is True:
        # PCA m=11
        P = PCA(DTR, m=11)
        DTR_PCA = numpy.dot(P.T, DTR)
        DTE_PCA = numpy.dot(P.T, DTE)

        PCA_mvg, PCA_mvg_naive, PCA_mvg_t, PCA_mvg_nt = compute_MVG_score(
            DTE_PCA,
            DTR_PCA,
            LTR,
            PCA_mvg,
            PCA_mvg_naive,
            PCA_mvg_t,
            PCA_mvg_nt,
            MVG_labels)

        # PCA m=10
        P = PCA(DTR, m=10)
        DTR_PCA = numpy.dot(P.T, DTR)
        DTE_PCA = numpy.dot(P.T, DTE)

        PCA2_mvg, PCA2_mvg_naive, PCA_2mvg_t, PCA2_mvg_nt = compute_MVG_score(
            DTE_PCA,
            DTR_PCA,
            LTR,
            PCA2_mvg,
            PCA2_mvg_naive,
            PCA2_mvg_t,
            PCA2_mvg_nt,
            MVG_labels)

    # π = 0.5 (our application prior)
    evaluation("minDCF: π=0.5", 0.5, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels, appendToTitle + 'minDCF_π=0.5__')

    ###############################

    # π = 0.1
    evaluation("minDCF: π=0.1", 0.1, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels, appendToTitle + 'minDCF_π=0.1__')

    ###############################

    # π = 0.9
    evaluation("minDCF: π=0.9", 0.9, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels, appendToTitle + "minDCF_π=0.9__")

    if PCA_Flag is True:
        #! PCA m=11
        # π = 0.5 (our application prior)
        evaluation("minDCF: π=0.5 | PCA m=11", 0.5, PCA_mvg,
                   PCA_mvg_naive,
                   PCA_mvg_t,
                   PCA_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.5_PCA m=11__")

        ###############################

        # π = 0.1
        evaluation("minDCF: π=0.1 | PCA m=11", 0.1, PCA_mvg,
                   PCA_mvg_naive,
                   PCA_mvg_t,
                   PCA_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.1_PCA m=11__")

        ###############################

        # π = 0.9
        evaluation("minDCF: π=0.9 | PCA m=11", 0.9, PCA_mvg,
                   PCA_mvg_naive,
                   PCA_mvg_t,
                   PCA_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.9_PCA m=11__")

        #! PCA m=10
        # π = 0.5 (our application prior)
        evaluation("minDCF: π=0.5 | PCA m=10", 0.5, PCA2_mvg,
                   PCA2_mvg_naive,
                   PCA2_mvg_t,
                   PCA2_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.5_PCA m=10__")

        ###############################

        # π = 0.1
        evaluation("minDCF: π=0.1 | PCA m=9", 0.1, PCA2_mvg,
                   PCA2_mvg_naive,
                   PCA2_mvg_t,
                   PCA2_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.1_PCA m=10__")

        ###############################

        # π = 0.9
        evaluation("minDCF: π=0.9 | PCA m=10", 0.9, PCA2_mvg,
                   PCA2_mvg_naive,
                   PCA2_mvg_t,
                   PCA2_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.9_PCA m=10__")


if __name__ == "__main__":
    
    #load and randomize TRAINING set
    DTR, LTR = load("dataset/Train.txt")
    DTR, LTR = randomize(DTR, LTR)
    
    #load and randomize TEST set
    DTE, LTE = load("dataset/Test.txt")
    DTE, LTE = randomize(DTE, LTE)

    D_merged, L_merged, idxTR_merged, idxTE_merged = split_db_after_merge(DTR, DTE, LTR, LTE) # Merged
    
    print("############    MVG    ##############")
    #evaluation_MVG(D_merged, L_merged, DTE, LTE, 'RAW_')                           #RAW features 
    evaluation_MVG(D_merged, L_merged, DTE, LTE, 'GAUSSIANIZED_', Gauss_flag=True) #Gaussianized features
    #evaluation_MVG(D_merged, L_merged, DTE, LTE, 'ZNORM_', zscore=True)            #Z-normed features
    
    