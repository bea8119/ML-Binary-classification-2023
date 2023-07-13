import sys
import numpy as np
from utils import *
from GMM_func import *
from GMM_main import *
from evaluator import *
from prettytable import PrettyTable


def validation_GMM(title, pi, GMM_llrs, LTE):
    GMM_llrs = np.hstack(GMM_llrs)
    llrs_tot = compute_min_DCF(GMM_llrs, LTE, pi, 1, 1)
    llrs_tot_act = compute_act_DCF(GMM_llrs, LTE, pi, 1, 1)
    llrs_tot_xvd = compute_act_DCF(GMM_llrs, LTE, pi, 1, 1, -np.log(pi / (1 - pi)))

    # t = PrettyTable(["Type", "minDCF"])
    # t.title = title
    # t.add_row(["GMM_EM", round(llrs_tot, 3)])
    # print(t)
    return round(llrs_tot, 3), round(llrs_tot_act, 3), round(llrs_tot_xvd, 3)


def ll_GMM(D, L, Dte, Lte, llr, cov, comp):
    # GMM_llrs, 'full', comp, i
    # CLASS PRIORS: WE CONSIDER A BALANCED APPLICATION

    # GMM MODELS
    # π = 0.5

    # optimal_m = 10
    optimal_comp = comp
    optimal_cov = cov
    optimal_alpha = 0.1
    optimal_psi = 0.01

    llr.extend(GMM_Full(D, Dte, L, optimal_alpha, 2 ** optimal_comp, optimal_cov).tolist())
    return llr


def plot_minDCF_GMM_eval(score_raw, score_gauss, title, components):
    labels = np.exp2(components).astype(int)
    raw_val, raw_eval = score_raw
    gauss_val, gauss_eval = score_gauss

    for i in range(components):
        labels.append(2 ** (i+1))

        x = np.arange(len(labels))  # the label locations
        width = 0.1  # the width of the bars
        plt.bar(x - 0.15, raw_val, width, label='Raw [val]', edgecolor='black', color='tab:orange', alpha=0.5)
        plt.bar(x - 0.05, raw_eval, width, label='Raw [eval]', edgecolor='black', color='tab:orange')
        plt.bar(x + 0.05, gauss_val, width, label='Gauss [val]', edgecolor='black', color='r', alpha=0.5)
        plt.bar(x + 0.05, gauss_eval, width, label='Gauss [eval]', edgecolor='black', color='r')

        plt.xticks(x, labels)
        plt.ylabel("DCF")
        plt.title(title)
        plt.legend()
        plt.savefig('./images/GMM/' + title)
        plt.show()


def print_minDCF_tables(score_raw, score_gauss, components):
    types = ['full-cov', 'diag-cov', 'tied full-cov', 'tied diag-cov']

    header = ['']
    n_comp = len(components)
    print(np.shape(score_raw))
    print(score_raw)
    score_raw = np.reshape(np.hstack(score_raw), ((n_comp), 4)).T
    score_gauss = np.reshape(np.hstack(score_gauss), ((n_comp), 4)).T

    comp = np.exp2(components).astype(int).tolist()

    print(np.shape(score_raw))
    for i in comp:
        header.append(i)
    print(header)
    for i in range(len(types)):
        t1 = PrettyTable(header)

        t1.title = types[i]

        raw_full = score_raw[i].tolist()
        gauss_full = score_gauss[i].tolist()

        raw_full.insert(0, 'raw')
        gauss_full.insert(0, 'gaussianized')
        t1.add_row(raw_full)
        t1.add_row(gauss_full)
        print(t1)
        # plot_minDCF_GMM(score_raw[i].tolist(), score_gauss[i].tolist(), types[i], components)


def print_act_DCF_tables(score_raw, score_gauss, components):
    types = ['full-cov', 'diag-cov', 'tied full-cov', 'tied diag-cov']

    header = ['']
    n_comp = len(components)
    print(np.shape(score_raw))
    print(score_raw)
    score_raw = np.reshape(np.hstack(score_raw), ((n_comp), 4)).T
    score_gauss = np.reshape(np.hstack(score_gauss), ((n_comp), 4)).T

    comp = np.exp2(components).astype(int).tolist()

    print(np.shape(score_raw))
    for i in comp:
        header.append(i)
    print(header)
    for i in range(len(types)):
        t1 = PrettyTable(header)

        t1.title = types[i]

        raw_full = score_raw[i].tolist()
        gauss_full = score_gauss[i].tolist()

        raw_full.insert(0, 'raw')
        gauss_full.insert(0, 'gaussianized')
        t1.add_row(raw_full)
        t1.add_row(gauss_full)
        print(t1)


def plot_minDCF_GMM(score_raw, score_gauss, title, components):
    labels = np.exp2(components).astype(int)

    # for i in range(components):
    #     labels.append(2 ** (i+1))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    plt.bar(x - 0.2, score_raw, width, label='Raw')
    plt.bar(x + 0.2, score_gauss, width, label='Gaussianized')

    plt.xticks(x, labels)
    plt.ylabel("DCF")
    plt.title(title)
    plt.legend()
    plt.savefig('../images/GMM/' + title)
    plt.show()


def evaluation_GMM(DTR, LTR, DTE, LTE, pi, comp, zscore=False, Gauss_flag=False):
    GMM_llrs = []
    GMM_llrsn = []
    GMM_llrst = []
    GMM_llrsnt = []
    
    GMM_labels = []

    D = DTR
    L = LTR

    Dte = DTE
    Lte = LTE

    if (zscore):
        D = scipy.stats.zscore(D, axis=1)
    if(Gauss_flag):
            D_training = D
            D = gaussianize_features(D, D)
            Dte = gaussianize_features(D_training, Dte)

    print("components: " + str(comp))
    GMM_labels = np.append(GMM_labels, Lte)
    GMM_labels = np.hstack(GMM_labels)

    # RAW DATA

    # full-cov
    GMM_llrs = ll_GMM(D, L, Dte, Lte, GMM_llrs, 'full', comp)
    

    # diag-cov
    GMM_llrsn = ll_GMM(D, L, Dte, Lte, GMM_llrsn, 'diag', comp)

    # full-cov tied
    GMM_llrst = ll_GMM(D, L, Dte, Lte, GMM_llrst, 'tied_full', comp)

    # diag-cov tied
    GMM_llrsnt = ll_GMM(D, L, Dte, Lte, GMM_llrsnt, 'tied_diag', comp)

    llrs_tot_min, llrs_tot_act, llrs_tot_xvd = validation_GMM("GMM full", pi, GMM_llrs, GMM_labels)
    llrsn_tot_min, llrsn_tot_act, llrsn_tot_xvd = validation_GMM("GMM diag", pi, GMM_llrsn, GMM_labels)
    llrst_tot_min, llrst_tot_act, llrst_tot_xvd = validation_GMM("GMM tied full", pi, GMM_llrst, GMM_labels)
    llrsnt_tot_min, llrsnt_tot_act, llrsnt_tot_xvd = validation_GMM("GMM tied diag", pi, GMM_llrsnt, GMM_labels)

    llrs_min = [llrs_tot_min, llrsn_tot_min, llrst_tot_min, llrsnt_tot_min]
    llrs_act = [llrs_tot_act, llrsn_tot_act, llrst_tot_act, llrsnt_tot_act]
    llrs_xvd = [llrs_tot_xvd, llrsn_tot_xvd, llrst_tot_xvd, llrsnt_tot_xvd]
    return llrs_min, llrs_act, llrs_xvd, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels


def bayes_error_min_act_plot_GMM(D, LTE, pi, title, ylim):
    p = numpy.linspace(-3, 3, 21)
    pylab.title(title)
    pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=False), color='r', label='actDCF')
    pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=True), 'r--', label='minDCF')
    pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=False, th=-np.log(pi / (1 - pi))), color='y', label='theoretical')
    pylab.ylim(0, ylim)
    pylab.legend()
    pylab.savefig('../images/DCF_' + title + '.png')
    pylab.show()


def bayes_plot_bestGMM(title, pi, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels):
    bayes_error_min_act_plot_GMM(GMM_llrs, GMM_labels, pi, 'GMM_full', 0.4)
    bayes_error_min_act_plot_GMM(GMM_llrsn, GMM_labels, pi, 'GMM_diag', 0.4)
    bayes_error_min_act_plot_GMM(GMM_llrst, GMM_labels, pi, 'GMM_tied', 0.4)
    bayes_error_min_act_plot_GMM(GMM_llrsnt, GMM_labels, pi, 'GMM_tied_diag', 0.4)


def evaluation_GMM_ncomp(DTR, LTR, DTE, LTE, pi, n, zscore=False, Gauss_flag=False):
    
    raw_min, raw_act, raw_x, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels = evaluation_GMM(
        DTR, LTR, DTE, LTE, pi, n, zscore, Gauss_flag=False)
    gauss_min, gauss_act, gauss_x, GMM_llrs_g, GMM_llrsn_g, GMM_llrst_g, GMM_llrsnt_g, GMM_labels_g = evaluation_GMM(DTR, LTR, DTE, LTE, pi, n, zscore=zscore, Gauss_flag=True)
    print(raw_act, raw_x)

    types = ['full-cov', 'diag-cov', 'tied full-cov', 'tied diag-cov']
    t = PrettyTable(["", 'minDCF'])
    t.title = "GMM π=" + str(pi)
    for i in range(len(raw_min)):
        t.add_row(["raw" + " " + types[i], raw_min[i],  raw_act[i], raw_x[i]])
    for i in range(len(gauss_min)):
        t.add_row(['gaussian. ' + types[i], gauss_min[i], gauss_act[i], gauss_x[i]])
    print(t)
    
   

    raw_llr = [GMM_llrs_raw, GMM_llrsn_raw, GMM_llrst_raw, GMM_llrsnt_raw, GMM_labels_raw]
    gauss_llr = [GMM_llrs_g, GMM_llrsn_g, GMM_llrst_g, GMM_llrsnt_g, GMM_labels_g]
    return GMM_llrst_raw, GMM_labels_raw

    return GMM_llrst
    # bayes_plot_bestGMM("prova", 0.5, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels)
    # plot_ROC(GMM_llrs, GMM_labels, 'GMM_full2')
    # plot_ROC(GMM_llrsn, GMM_labels, 'GMM_diag2')
    # plot_ROC(GMM_llrst, GMM_labels, 'GMM_tied2')
    # plot_ROC(GMM_llrsnt, GMM_labels, 'GMM_tied_diag2')


def evaluation_scores_GMM_ncomp(typeof, DTR, LTR, DTE, LTE, pi, n, zscore=False):
    raw_min, raw_act, raw_x, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels = evaluation_GMM(
        DTR, LTR, DTE, LTE, pi, n, zscore)

    return GMM_llrst

def evaluation_GMM_tot(DTR, LTR, DTE, LTE, pi, zscore=False, gauss=False):
    score_raw_min = []
    score_gauss_min = []
    
    score_raw_act = []
    score_gauss_act = []
    
    score_raw_xvd = []
    score_gauss_xvd = []
    # We'll train from 1 to 2^4 components
    
    # We'll train from 1 to 2^4 components
    componentsToTry=[1,2,3,4,5]
    for comp in componentsToTry:

        print('RAW DATA')
        raw_min, raw_act, raw_xvd, *_ = evaluation_GMM(
        DTR, LTR, DTE, LTE, pi, comp, zscore, Gauss_flag=False )
        score_raw_min.append(raw_min)
        score_raw_act.append(raw_act)
        score_raw_xvd.append(raw_xvd)

        print('GAUSSIANIZED')
        gauss_min, gauss_act, gauss_xvd, *_ = evaluation_GMM(
        DTR, LTR, DTE, LTE, pi, comp, zscore, Gauss_flag=True )
        score_gauss_min.append(gauss_min)
        score_gauss_act.append(gauss_act)
        score_gauss_xvd.append(gauss_xvd)
    print("======= min DCF =======")
    print_minDCF_tables(score_raw_min, score_gauss_min, componentsToTry)
    #print("======= act DCF =======")
   # print_act_DCF_tables(score_raw_act, score_gauss_act, componentsToTry)
   # print("======= theoretical =======")
   # print_act_DCF_tables(score_raw_xvd, score_gauss_xvd, componentsToTry)
    plot_minDCF_GMM_eval(score_raw_min, score_gauss_min, "GMMEval", componentsToTry)

if __name__ == "__main__":
    
    #load and randomize TRAINING set
    DTR, LTR = load("dataset/Train.txt")
    DTR, LTR = randomize(DTR, LTR)
    
    #load and randomize TEST set
    DTE, LTE = load("dataset/Test.txt")
    DTE, LTE = randomize(DTE, LTE)

    D_merged, L_merged, idxTR_merged, idxTE_merged = split_db_after_merge(DTR, DTE, LTR, LTE) # Merged
    
    
    print("############    Gaussian Mixture Models   ##############")
    evaluation_GMM_tot(D_merged, L_merged, DTE, LTE, 0.5)
    
    #evaluation_GMM_ncomp(D_merged, L_merged, DTE, LTE, 0.5, 4)
    #evaluation_GMM_ncomp(D_merged, L_merged, DTE, LTE, 0.1, 4 )
    #evaluation_GMM_ncomp(D_merged, L_merged, DTE, LTE, 0.9, 4 )
    
    
   


