import DCF
import numpy as np
import utils as u
from plotting import DET_curves, ROC_curves, bayes_error_plots
from matplotlib.pyplot import show, savefig, figure

saveFigure = True

_, LTR = u.load('./dataset/Train.txt')
np.random.seed(0)
idx = np.random.permutation(LTR.shape[0])
trueL_ordered = LTR[idx]

# DET and ROC curves on Training set
# scores_list = [
#    np.load('./data_npy/scores_MVG_K_fold_PCA_None.npy'),
#    np.load('./data_npy/scores_SVM_K_fold_PCA_None.npy'),
#    np.load('./data_npy/scores_Fusion_K_fold_PCA_None.npy'),
# ]

csf_names = [
   #'MVG',
   'SVM',
   #'Fusion',
]

FNR_list = []
FPR_list = []
TPR_list = []

print('Calculating ROC and DET plot values...')

# for scores in scores_list:
#    FNR, FPR, TPR = DCF.ROC_DET_arrays(scores, trueL_ordered)
#    FNR_list.append(FNR)
#    FPR_list.append(FPR)
#    TPR_list.append(TPR)

# print('Plotting...')
# roc_fig = ROC_curves(FPR_list, TPR_list, csf_names)
# det_fig = DET_curves(FPR_list, FNR_list, csf_names)
# show()
# print('Plotting done.')

# if saveFigure:
#    figure(roc_fig)
#    savefig('./images/ROC_Training.png')
#    figure(det_fig)
#    savefig('./images/DET_Training.png')
#    print('Saved.')
# print()

# Bayes error plots for training set
print('Bayes Error plots part starting...')
effPriorLogOdds = np.linspace(-3, 3, 21) # Common for all the plots

# For showing lack of calibration (minDCF and actualDCF) of MVG and SVM

# dcf_list = []
# print('Calculating DCF and minDCF for uncalibrated MVG...')
# dcf_norm, dcf_min = DCF.DCF_vs_priorLogOdds(effPriorLogOdds, np.load('./data_npy/scores_MVG_K_fold_PCA_None.npy'), trueL_ordered)
# print('Done.\n')
# dcf_list.append(dcf_norm)
# dcf_list.append(dcf_min)
# print('Calculating DCF and minDCF for uncalibrated SVM...')
# dcf_norm, dcf_min = DCF.DCF_vs_priorLogOdds(effPriorLogOdds, np.load('./data_npy/scores_SVM_K_fold_PCA_None.npy'), trueL_ordered)
# print('Done.\n')
# dcf_list.append(dcf_norm)
# dcf_list.append(dcf_min)
# param_list = [
#   ('MVG actDCF', False, 'r'),
#   ('MVG minDCF', True, 'r'),
#   ('SVM actDCF', False, 'b'),
#   ('SVM minDCF', True, 'b'),
# ]

# print('Bayes error plotting...')
# bayes_fig = bayes_error_plots(effPriorLogOdds, dcf_list, param_list, 'no calibration')
# print('Plotting done.\n')

# if savefig:
#   figure(bayes_fig)
#   savefig('./images/Bayes_Train_MVG_SVM_uncalibrated.png')
#   print('Saved.')

# For showing effects of calibration (minDCF and actualDCF) of MVG and SVM
dcf_list = []
print('Calculating DCF and minDCF for calibrated MVG...')
# dcf_norm, dcf_min = DCF.DCF_vs_priorLogOdds(effPriorLogOdds, np.load('./data_npy/scores_MVG_K_fold_PCA_None_calibrated.npy'), trueL_ordered)
# print('Done.\n')
# dcf_list.append(dcf_norm)
# dcf_list.append(dcf_min)
print('Calculating DCF and minDCF for calibrated SVM...')
dcf_norm, dcf_min = DCF.DCF_vs_priorLogOdds(effPriorLogOdds, np.load('./data_npy/scores_SVM_K_fold_PCA_None_calibrated.npy'), trueL_ordered)
print('Done.\n')
dcf_list.append(dcf_norm)
dcf_list.append(dcf_min)
param_list = [
    #('MVG actDCF (cal)', False, 'r'),
    #('MVG minDCF (cal)', True, 'r'),
    ('SVM actDCF (cal)', False, 'b'),
    ('SVM minDCF (cal)', True, 'b'),
]
print('Bayes error plotting...')
bayes_fig = bayes_error_plots(effPriorLogOdds, dcf_list, param_list, 'calibration')
print('Plotting done.\n')

if savefig:
    figure(bayes_fig)
    savefig('./images/Bayes_Train_MVG_SVM_calibrated.png')
    print('Saved.')

# # For showing calibrated scores's actual DCF and Fusion model min and actual DCFs
# dcf_list = []
# dcf_norm, _ = DCF.DCF_vs_priorLogOdds(effPriorLogOdds, np.load('./data_npy/scores_MVG_K_fold_PCA_None_calibrated.npy'), trueL_ordered, actualOnly=True)
# dcf_list.append(dcf_norm)
# dcf_norm, _ = DCF.DCF_vs_priorLogOdds(effPriorLogOdds, np.load('./data_npy/scores_SVM_K_fold_PCA_None_calibrated.npy'), trueL_ordered, actualOnly=True)
# dcf_list.append(dcf_norm)
# dcf_norm, dcf_min = DCF.DCF_vs_priorLogOdds(effPriorLogOdds, np.load('./data_npy/scores_Fusion_K_fold_PCA_None.npy'), trueL_ordered)
# dcf_list.append(dcf_norm)
# dcf_list.append(dcf_min)
# param_list = [
#     ('MVG actDCF (cal)', False, 'r'),
#     ('SVM actDCF (cal)', False, 'b'),
#     ('Fusion actDCF', False, 'g'),
#     ('Fusion minDCF', True, 'g'),
# ]
# bayes_fig = bayes_error_plots(effPriorLogOdds, dcf_list, param_list, 'fusion')

# if savefig:
#     figure(bayes_fig)
#     savefig('./image/Bayes_Train_MVG_SVM_Fusion.png')
#     print('Saved.')

show()