from utils import * 
from evaluator import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn

CLASS_NAMES = ['Male', 'Female']
ATTRIBUTE_NAMES = ['1', '2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

def plotHistogram(D, L, class_names, attribute_names):
    # dataset for each class
    d0 = D[:, L == 0]
    d1 = D[:, L == 1]
    
    n_features = D.shape[0]
    
    alpha = 0.6 # Opacity coefficient
    bins = 70 # N_bins

    # now, divide per value type in a loop and plot a figure containing all the classes

    for i in range(n_features):
        plt.figure(attribute_names[i])
        plt.hist(d0[i, :], bins=bins, density=True, alpha=alpha, label=class_names[0], color='r', ec='black')
        plt.hist(d1[i, :], bins=bins, density=True, alpha=alpha, label=class_names[1], color='b', ec='black')
        plt.xlabel(attribute_names[i])
        plt.legend()
        plt.tight_layout()
        plt.grid(visible=True)
    # plt.show()

def plotHeatmap(D, L):
    plt.figure('Whole Dataset')
    seaborn.heatmap(np.abs(np.corrcoef(D)), linewidth=0.5, cmap="Greys", square=True, cbar=False)
    plt.xlabel('Whole Dataset')
    plt.figure('Male class samples')
    seaborn.heatmap(np.abs(np.corrcoef(D[:, L == 0])), linewidth=0.5, cmap="Reds", square=True, cbar=False)
    plt.xlabel('Male Class Samples')
    plt.figure('Female class samples')
    seaborn.heatmap(np.abs(np.corrcoef(D[:, L == 1])), linewidth=0.5, cmap="Blues", square=True, cbar=False)
    plt.xlabel('Female Class Samples')
    # plt.show()

def plotDCFmin_vs_lambda(l_arr, min_DCF_single_arr, min_DCF_kfold_arr, m_PCA, n, K, colors, eff_priors, quad=False, save_fig=False):
    '''Receives 3 arrays to plot (curves) for each eff_prior'''
    if min_DCF_single_arr:
        fig_name_single = 'Single-fold ({}-to-1) {} Log Reg -> lambda tuning {}'.format(
            n, 'Quadratic' if quad else 'Linear', '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_single = plt.figure(fig_name_single)
    if min_DCF_kfold_arr:
        fig_name_kfold = '{}-fold {} Log Reg -> lambda tuning {}'.format(
            K, 'Quadratic' if quad else 'Linear', '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_kfold = plt.figure(fig_name_kfold)

    for i in range(len(eff_priors)):
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.plot(l_arr, min_DCF_single_arr[i], color=colors[i], label='min DCF {} = {}'.format(r'$\tilde{\pi}$', eff_priors[i][0]))
            plt.xlim([min(l_arr), max(l_arr)])
            plt.xscale('log')
            plt.xlabel(r'$\lambda$')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.plot(l_arr, min_DCF_kfold_arr[i], color=colors[i], label='min DCF {} = {}'.format(r'$\tilde{\pi}$', eff_priors[i][0]))
            plt.xlim([min(l_arr), max(l_arr)])
            plt.xscale('log')
            plt.xlabel(r'$\lambda$')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
    
    if save_fig:
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.savefig('../plots/' + fig_name_single.replace('>', '-') + '.png')
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.savefig('../plots/' + fig_name_kfold.replace('>', '-') + '.png')

def plotDCFmin_vs_lambda_eval(l_arr, minDCF_arr, priorT, colors, eff_priors, quad=False, saveFig=False):
    ''' Tuning of lambda hyperparameter alone, on every application point '''
    fig_name = 'Test set {} LogReg -> l tuning (pi_T = {})'.format('Quadratic' if quad else 'Linear', priorT)
    plt.figure(fig_name)

    for i in range(len(eff_priors)):
        if minDCF_arr:
            plt.plot(l_arr, minDCF_arr[i], color=colors[i], label='min DCF {} = {}'.format(r'$\tilde{\pi}$', eff_priors[i][0]))
            plt.xlim([min(l_arr), max(l_arr)])
            plt.xscale('log')
            plt.xlabel(r'$\lambda$')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
    if saveFig:
        plt.savefig('./images/' + fig_name.replace('>', '-') + '.png')

def plotDCFmin_vs_C_linearSVM(C_arr, min_DCF_single_arr, min_DCF_kfold_arr, pi_b, m_PCA, n, K, colors, eff_priors, save_fig=False):
    '''Tuning of C parameter alone, on every application point'''
    if min_DCF_single_arr:
        fig_name_single = 'Single-fold ({}-to-1) Linear SVM -> C tuning (pi_T = {}) {}'.format(
            n, 'unbalanced' if pi_b is None else pi_b, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_single = plt.figure(fig_name_single)
    if min_DCF_kfold_arr:
        fig_name_kfold = '{}-fold Linear SVM -> C tuning (pi_T = {}) {}'.format(
            K, 'unbalanced' if pi_b is None else pi_b, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_kfold = plt.figure(fig_name_kfold)

    for i in range(len(eff_priors)):
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.plot(C_arr, min_DCF_single_arr[i], color=colors[i], label='min DCF {} = {}'.format(r'$\tilde{\pi}$', eff_priors[i][0]))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.plot(C_arr, min_DCF_kfold_arr[i], color=colors[i], label='min DCF {} = {}'.format(r'$\tilde{\pi}$', eff_priors[i][0]))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)

    if save_fig:
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.savefig('./images/' + fig_name_single.replace('>', '-') + '.png')
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.savefig('./images/' + fig_name_kfold.replace('>', '-') + '.png')

def plotDCFmin_vs_C_linearSVM_eval(C_arr, minDCF_arr, pi_b, colors, app_points, saveFig=False):
    ''' Tuning of C alone for linear SVM (using the test set) '''
    fig_name = 'Test set Linear SVM -> C tuning (pi_T = {})'.format('unbalanced' if pi_b is None else pi_b)
    plt.figure(fig_name)
    for i in range(len(app_points)):
        plt.plot(C_arr, minDCF_arr[i], color=colors[i], label='min DCF {} = {}'.format(r'$\tilde{\pi}$', app_points[i][0]))
        plt.xlim([min(C_arr), max(C_arr)])
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('DCF')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(visible=True)
    if saveFig:
        plt.savefig('./images/' + fig_name.replace('>', '-') + '.png')


def plotDCFmin_vs_C_quadSVM(C_arr, min_DCF_single_arr, min_DCF_kfold_arr, m_PCA, n, K, colors, app_point, c_list, save_fig=False):
    '''Tuning of C jointly with c (in linear scale), take different values of c on the same application point'''
    if min_DCF_single_arr:
        fig_name_single = 'Single-fold ({}-to-1) Quadratic Kernel SVM -> C - c tuning {}'.format(
            n, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_single = plt.figure(fig_name_single)
    if min_DCF_kfold_arr:
        fig_name_kfold = '{}-fold Quadratic Kernel SVM -> C - c tuning {}'.format(
            K, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_kfold = plt.figure(fig_name_kfold)
    
    for i in range(len(c_list)):
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.plot(C_arr, min_DCF_single_arr[i], color=colors[i], label='min DCF {} = {}, c = {}'.format(
                r'$\tilde{\pi}$', app_point[0], c_list[i]))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.plot(C_arr, min_DCF_kfold_arr[i], color=colors[i], label='min DCF {} = {}, c = {}'.format(
                r'$\tilde{\pi}$', app_point[0], c_list[i]))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)

    if save_fig:
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.savefig('./images/' + fig_name_single.replace('>', '-') + '.png')
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.savefig('./images/' + fig_name_kfold.replace('>', '-') + '.png')

def plotDCFmin_vs_C_quadSVM_eval(C_arr, minDCF_arr, pi_b, colors, app_point, c_list, saveFig=False):
    ''' Tuning of C and c jointly for quad SVM (using the test set) '''
    fig_name = 'Test set Quad SVM -> C and c tuning (pi_T = {})'.format('unbalanced' if pi_b is None else pi_b)
    plt.figure(fig_name)
    for i in range(len(c_list)):
        plt.figure(fig_name)
        plt.plot(C_arr, minDCF_arr[i], color=colors[i], label='min DCF {} = {}, c = {}'.format(
            r'$\tilde{\pi}$', app_point[0], c_list[i]))
        plt.xlim([min(C_arr), max(C_arr)])
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('DCF')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(visible=True)
    if saveFig:
        plt.savefig('./images/' + fig_name.replace('>', '-') + '.png')

def plotDCFmin_vs_C_RBFSVM(C_arr, min_DCF_single_arr, min_DCF_kfold_arr, m_PCA, n, K, colors, app_point, gamma_list, save_fig=False):
    '''Tuning of C jointly with gamma (in log scale), take different values of gamma on the same application point'''
    if min_DCF_single_arr:
        fig_name_single = 'Single-fold ({}-to-1) RBF Kernel SVM -> C - gamma tuning {}'.format(
            n, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_single = plt.figure(fig_name_single)
    if min_DCF_kfold_arr:
        fig_name_kfold = '{}-fold RBF Kernel SVM -> C - gamma tuning {}'.format(
            K, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_kfold = plt.figure(fig_name_kfold)
    
    for i in range(len(gamma_list)):
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.plot(C_arr, min_DCF_single_arr[i], color=colors[i], label='min DCF {} = {}, log {} = {}'.format(
                r'$\tilde{\pi}$', app_point[0], r'$\gamma$', int(np.log10(gamma_list[i]))))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.plot(C_arr, min_DCF_kfold_arr[i], color=colors[i], label='min DCF {} = {}, log {} = {}'.format(
                r'$\tilde{\pi}$', app_point[0], r'$\gamma$', int(np.log10(gamma_list[i]))))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)

    if save_fig:
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.savefig('./images/' + fig_name_single.replace('>', '-') + '.png')
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.savefig('./images/' + fig_name_kfold.replace('>', '-') + '.png')

def plotDCFmin_vs_C_RBFSVM_eval(C_arr, minDCF_arr, pi_b, colors, app_point, gamma_list, saveFig=False):
    ''' Tuning of C and gamma jointly for RBF kernel SVM (using the test set) '''
    fig_name = 'Test set RBF SVM -> C and gamma tuning (pi_T = {})'.format('unbalanced' if pi_b is None else pi_b)
    plt.figure(fig_name)
    for i in range(len(gamma_list)):
        plt.figure(fig_name)
        plt.plot(C_arr, minDCF_arr[i], color=colors[i], label='min DCF {} = {}, log {} = {}'.format(
            r'$\tilde{\pi}$', app_point[0], r'$\gamma$', int(np.log10(gamma_list[i]))))
        plt.xlim([min(C_arr), max(C_arr)])
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('DCF')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(visible=True)
    if saveFig:
        plt.savefig('./images/' + fig_name.replace('>', '-') + '.png')

def create_GMM_figure(tied, diag):
    '''Receives tied and diag flags and n=4, returns a list of figure objects with the appropriate names'''
    GMM_type = ''
    if tied:
        GMM_type += 'Tied '
    if diag:
        GMM_type += 'Diag '
    if not diag and not tied:
        GMM_type += 'Full '
    GMM_type += 'Covariance '

    return plt.figure(GMM_type)

def plotGMM(n_splits, dcf_min_list, eff_prior, tied_diag_pairs, colors, PCA_list):

    one = 1

    for i, (tied, diag) in enumerate(tied_diag_pairs):
        GMM_type = ''
        if tied:
            GMM_type += 'Tied '
        if diag:
            GMM_type += 'Diag '
        if not diag:
            GMM_type += 'Full '
        GMM_type += 'Covariance '

        plt.figure('{}GMM classifier'.format(GMM_type))
        for j, m in enumerate(PCA_list):
            plt.bar(np.arange(1, n_splits + 1) + 0.1 * one, dcf_min_list[j][i], label='min DCF {} = {} {}'.format(
                r'$\tilde{\pi}$', eff_prior, '(no PCA)' if m is None else '(PCA m = {})'.format(m)),
                color=colors[j], width=0.2)
            one *= -1
        plt.xlabel('Number of components')
        plt.xticks(np.arange(1, n_splits + 1), 2**np.arange(1, n_splits + 1))
        plt.ylabel('DCF')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(visible=True)
    
#########################################################################################################
def bayes_error_plot(pArray, scores, labels, minCost=False, th=None):
    y = []
    for p in pArray:
        pi = 1.0 / (1.0 + np.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1, th))
    return np.array(y)


def bayes_error_plot_compare(pi, scores, labels):
    y = []
#    pi = 1.0 / (1.0 + np.exp(-pi)) 
    y.append(compute_min_DCF(scores, labels, pi, 1, 1))
    return np.array(y)

def plot_DCF(x, y, xlabel, title, base=10):
    plt.figure()
    plt.plot(x, y[0], label= 'min DCF prior=0.5', color='b')
    plt.plot(x, y[1], label= 'min DCF prior=0.9', color='g')
    plt.plot(x, y[2], label= 'min DCF prior=0.1', color='r')
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base=base)
    plt.legend([ "min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig('./images/DCF_' + title+ '.svg')
    plt.show()
    return


def plot_ROC(llrs, LTE, title):
    thresholds = numpy.array(llrs)
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])   #evaluate all possible thresholds in range [-infinte, s_1,...,s_m, +infinite] where s_1,s_m are the test scores! 
    
    FPR = numpy.zeros(thresholds.size)
    TPR = numpy.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = numpy.int32(llrs > t)                    #make label prediction
        conf = confusion_matrix_binary(Pred, LTE)
        TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
        FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
    
    pylab.plot(FPR, TPR)
    pylab.title(title)
    pylab.savefig('./images/ROC_' + title + '.png')
    pylab.show()


def ROC_curves(FPR_list, TPR_list, csf_names):
    fig = plt.figure('ROC')
    for FPR, TPR, name in zip(FPR_list, TPR_list, csf_names):
        plt.plot(FPR, TPR, label=name)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.legend(loc='best')
    return fig   

def DET_curves(FPR_list, FNR_list, csf_names):
    fig = plt.figure('DET')
    for FPR, FNR, name in zip(FPR_list, FNR_list, csf_names):
        plt.plot(FPR, FNR, label=name)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('FPR')
    plt.ylabel('FNR')
    plt.grid()
    plt.legend(loc='best')
    return fig












#########################################################################################
def bayes_error_plots(effPriors, DCF_list, param_list, title):
    fig = plt.figure('Bayes error plot - ' + title)
    for DCF, (label, dashed, color) in zip(DCF_list, param_list):
        plt.plot(effPriors, DCF, label=label, linestyle='dashed' if dashed else None, color=color)
    plt.xlim([-3, 3])
    plt.xlabel('Prior log-odds')
    plt.ylim([0, 1.1])
    plt.ylabel('DCF value')
    plt.grid()
    plt.legend(loc='best')
    return fig




def heatmap():

    # Pre-processing (Z-normalization)
    DTR, mean, std = znorm(DTR)
    # DTE = f.Z_normalization(DTE, mean, std)

    # Plot distribution of attribute values (after Z-Normalizing) for each class
    plotHistogram(DTR, LTR, CLASS_NAMES, ATTRIBUTE_NAMES)

    # Plot heatmap of covariance
    plotHeatmap(DTR, LTR)
    plt.show()
    plt.savefig()

#GMM plots
    

def print_minDCF_tables(score_raw, score_gauss, components):
    types = ['full-cov', 'diag-cov', 'tied full-cov', 'tied diag-cov']
    
    header = ['']
    n_comp = len(components)
    print(np.shape(score_raw))
    print(score_raw)
    score_raw = np.reshape(np.hstack(score_raw), (n_comp, 4)).T
    score_gauss = np.reshape(np.hstack(score_gauss), (n_comp, 4)).T
    
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
        
        raw_full.insert(0,'raw')
        gauss_full.insert(0,'gaussianized')
        t1.add_row(raw_full)
        t1.add_row(gauss_full)
        print(t1)
        plot_minDCF_GMM(score_raw[i].tolist(), score_gauss[i].tolist(), types[i], components)
             
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
        
        raw_full.insert(0,'raw')
        gauss_full.insert(0,'gaussianized')
        t1.add_row(raw_full)
        t1.add_row(gauss_full)
        print(t1)



def plot_minDCF_GMM(score_raw, score_gauss, title, components):
    labels = np.exp2(components).astype(int)
    
    # for i in range(components):
    #     labels.append(2 ** (i+1))
    
 
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    plt.bar(x - 0.2, score_raw, width, label = 'Raw')
    plt.bar(x + 0.2, score_gauss, width, label = 'Gaussianized')
      
    plt.xticks(x, labels)
    plt.ylabel("DCF")
    plt.title(title)
    plt.legend()
    plt.savefig('./images/GMM/' + title)
    plt.show()
    


def bayes_error_min_act_plot_GMM(D, LTE, pi, title, ylim):
    p = numpy.linspace(-3, 3, 21)
    pylab.title(title)
    pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=False), color='r', label='actDCF')
    pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=True), 'r--', label='minDCF')
    pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=False, th=-np.log(pi / (1-pi))), color='y', label='theoretical')
    pylab.ylim(0, ylim)
    pylab.legend()
    pylab.savefig('./images/DCF_' + title + '.png')
    pylab.show()
    
    

def bayes_plot_bestGMM(title, width, pi, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels):
    bayes_error_min_act_plot_GMM(GMM_llrs, GMM_labels, pi, title + 'GMM_full', width)
    bayes_error_min_act_plot_GMM(GMM_llrsn, GMM_labels, pi, title + 'GMM_diag', width)
    bayes_error_min_act_plot_GMM(GMM_llrst, GMM_labels, pi, title + 'GMM_tied', width)
    bayes_error_min_act_plot_GMM(GMM_llrsnt, GMM_labels, pi, title + 'GMM_tied_diag', width)
'''
def ROC_GMM():
 '''   
   


if __name__ == '__main__':
    DTR, LTR = load('./dataset/Train.txt')
    DTE, LTE = load('./dataset/Test.txt')
    DTR_GAUSS = gaussianize_features(DTR, DTR)