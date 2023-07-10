

import sys
import numpy as np
from MVG_func import *


from utils import *
from prettytable import PrettyTable
from plotting import *
from evaluator import *


def evaluation(title, pi, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels, appendToTitle):
    MVG_res = np.hstack(MVG_res)
    MVG_naive = np.hstack(MVG_naive)
    MVG_t = np.hstack(MVG_t)
    MVG_nt = np.hstack(MVG_nt)
    
    #minimum DCFs for each model
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


def compute_MVG_score(Dte, D, L, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels):
    _, _, llrs = MVG(Dte, D, L)             #log-likelihood ratio multivariate gausian for each sample
    _, _, llrsn = naive_MVG(Dte, D, L)      #log-likelihood ratio- naive for each sample
    _, _, llrst = tied_cov_GC(Dte, D, L)    #log-likelihood ratio- tied cov. for each sample
    _, _, llrsnt = tied_cov_naive_GC(Dte, D, L)   #log-likelihood ratio- naive+tied cov. for each sample

    MVG_res.append(llrs)
    MVG_naive.append(llrsn)
    MVG_t.append(llrst)
    MVG_nt.append(llrsnt)
    
    return MVG_res, MVG_naive, MVG_t, MVG_nt

def validation_MVG(DTR, LTR, appendToTitle, PCA_Flag=True, Gauss_flag = False, zscore=False):
    
    MVG_labels = []
    
    #no PCA
    MVG_res = []        #full covariance
    MVG_naive = []      #naive 
    MVG_t = []          #tied covarinace
    MVG_nt = []         #naive + tied
    
    #first PCA (m=11)
    PCA_mvg = []         
    PCA_mvg_naive = []
    PCA_mvg_t = []
    PCA_mvg_nt = []

    #second PCA (m=10)
    PCA2_mvg = []
    PCA2_mvg_naive = []
    PCA2_mvg_t = []
    PCA2_mvg_nt = []
    
    #! Kfold approach
    K = 5
    Dtr = np.split(DTR, K, axis=1)  
    Ltr = np.split(LTR, K)
    
    for i in range(K):                
        
        D = []                        
        L = []
        if i == 0:                            
            D.append(np.hstack(Dtr[i + 1:])) 
            L.append(np.hstack(Ltr[i + 1:]))
        elif i == K - 1:                       
            D.append(np.hstack(Dtr[:i]))
            L.append(np.hstack(Ltr[:i]))
        else:                                 
            D.append(np.hstack(Dtr[:i]))
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[:i]))
            L.append(np.hstack(Ltr[i + 1:]))

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = Dtr[i]                #i_th fold will be used for evaluation
        Lte = Ltr[i]

        if (zscore):
            D, Dte = znorm(D, Dte)

        if (Gauss_flag):
            D_training = D
            D = gaussianize_features(D, D)
            Dte = gaussianize_features(D_training, Dte)

        MVG_labels = np.append(MVG_labels, Lte, axis=0)
        MVG_labels = np.hstack(MVG_labels)
        
        
        # Once we have computed our folds, we can try different models

        MVG_res, MVG_naive, MVG_t, MVG_nt = compute_MVG_score(
            Dte,
            D,
            L,
            MVG_res,
            MVG_naive,
            MVG_t,
            MVG_nt,
            MVG_labels)

        if PCA_Flag is True:
            
            # PCA m=11
            P = PCA(D, L, m=11)
            DTR_PCA = np.dot(P.T, D)
            DTE_PCA = np.dot(P.T, Dte)

            PCA_mvg, PCA_mvg_naive, PCA_mvg_t, PCA_mvg_nt = compute_MVG_score(
                DTE_PCA,
                DTR_PCA,
                L,
                PCA_mvg,
                PCA_mvg_naive,
                PCA_mvg_t,
                PCA_mvg_nt,
                MVG_labels)

            # PCA m=10
            P = PCA(D, L, m=10)
            DTR_PCA = np.dot(P.T, D)
            DTE_PCA = np.dot(P.T, Dte)

            PCA2_mvg, PCA2_mvg_naive, PCA_2mvg_t, PCA2_mvg_nt = compute_MVG_score(
                DTE_PCA,
                DTR_PCA,
                L,
                PCA2_mvg,
                PCA2_mvg_naive,
                PCA2_mvg_t,
                PCA2_mvg_nt,
                MVG_labels)

    #evaluate with i_th fold
    
    # π = 0.5 (our application prior)
    evaluation("minDCF: π=0.5", 0.5, 
               MVG_res, 
               MVG_naive, 
               MVG_t, 
               MVG_nt, 
               MVG_labels, 
               appendToTitle + 'minDCF_π=0.5__')

    ###############################

    # π = 0.1
    evaluation("minDCF: π=0.1", 0.1, 
               MVG_res, 
               MVG_naive, 
               MVG_t, 
               MVG_nt, 
               MVG_labels, 
               appendToTitle + 'minDCF_π=0.1__')

    ###############################

    # π = 0.9
    evaluation("minDCF: π=0.9", 0.9, 
               MVG_res, 
               MVG_naive, 
               MVG_t, 
               MVG_nt, 
               MVG_labels, 
               appendToTitle + "minDCF_π=0.9__")

    if PCA_Flag is True:
        
        #! PCA m=11
        # π = 0.5 
        evaluation("minDCF: π=0.5 | PCA m=11", 0.5, 
                   PCA_mvg,
                   PCA_mvg_naive,
                   PCA_mvg_t,
                   PCA_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.5_PCA m=11__")

        # π = 0.1
        evaluation("minDCF: π=0.1 | PCA m=11", 0.1, 
                   PCA_mvg,
                   PCA_mvg_naive,
                   PCA_mvg_t,
                   PCA_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.1_PCA m=11__")

        
        # π = 0.9
        evaluation("minDCF: π=0.9 | PCA m=11", 0.9, 
                   PCA_mvg,
                   PCA_mvg_naive,
                   PCA_mvg_t,
                   PCA_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.9_PCA m=11__")
        
        #! PCA m=10
        # π = 0.5 
        evaluation("minDCF: π=0.5 | PCA m=10", 0.5, 
                   PCA2_mvg,
                   PCA2_mvg_naive,
                   PCA2_mvg_t,
                   PCA2_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.5_PCA m=10__")


        # π = 0.1
        evaluation("minDCF: π=0.1 | PCA m=10", 0.1, 
                   PCA2_mvg,
                   PCA2_mvg_naive,
                   PCA2_mvg_t,
                   PCA2_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.1_PCA m=10__")


        # π = 0.9
        evaluation("minDCF: π=0.9 | PCA m=10", 0.9, 
                   PCA2_mvg,
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
    
    print("############    MVG    ##############")
    validation_MVG(DTR, LTR, 'RAW_')                           #RAW features 
    validation_MVG(DTR, LTR, 'GAUSSIANIZED_', Gauss_flag=True) #Gaussianized features
    validation_MVG(DTR, LTR, 'ZNORM_', zscore=True)            #Z-normed features
    
    