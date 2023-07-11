import sys
import numpy

sys.path.append('../')
from utils import *


def confusion_matrix_binary(Lpred, LTE):
    C = numpy.zeros((2, 2))
    C[0, 0] = ((Lpred == 0) * (LTE == 0)).sum()
    C[0, 1] = ((Lpred == 0) * (LTE == 1)).sum()
    C[1, 0] = ((Lpred == 1) * (LTE == 0)).sum()
    C[1, 1] = ((Lpred == 1) * (LTE == 1)).sum()
    return C


def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th = -numpy.log(pi * Cfn) + numpy.log((1 - pi) * Cfp)
    P = scores > th
    return numpy.int32(P)


#! BINARY EMPIRICAL BAYES RISK (DCF)

# DCF= pi * C_fn *FNR + (1-pi) * Cfp * FPR 
def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp):
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])  #false negative rate
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])  #false positive rate
    return pi * Cfn * fnr + (1 - pi) * Cfp * fpr


def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp): 
    empBayes = compute_emp_Bayes_binary(CM, pi, Cfn, Cfp)
    return empBayes / min(pi * Cfn, (1 - pi) * Cfp) 


def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)    #predictions of labels
    CM = confusion_matrix_binary(Pred, labels)           #confusion matrix:  [TN    FN]
                                                         #                   [FP    TP]
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)# return the cost for this given set of predictions that generates a given number of False positives and False Negatives!

def compute_min_DCF(scores, labels, pi, Cfn, Cfp):  
    t = numpy.array(scores)    
    t.sort()                     
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
        
    return numpy.array(dcfList).min()   #return only the minimum between all computed DCF, corresponding to the best threshold for that application with pi





##########################################################################################################################
#!OLD DCF.PY FILE


def compute_confusion_matrix(predL, trueL, K):
	conf_matrix = np.zeros((K, K))
	for pCls, tCls in zip(predL, trueL):
		conf_matrix[pCls, tCls] += 1

	return conf_matrix


def compute_optBayes_decisions(scores, pi_1=None, C_fn=None, C_fp=None, given_threshold=None):
	if (given_threshold is None):
		threshold = - np.log((pi_1 * C_fn) / ((1 - pi_1) * C_fp))
	else:
		threshold = given_threshold
	return np.int32(scores > threshold)


def DCF_binary(confusion_m, pi_1, C_fn, C_fp):
	FNR = confusion_m[0, 1] / (confusion_m[0, 1] + confusion_m[1, 1])
	FPR = confusion_m[1, 0] / (confusion_m[0, 0] + confusion_m[1, 0])
	return pi_1 * C_fn * FNR + (1 - pi_1) * C_fp * FPR


def FNR_FPR(confusion_m):
	'''Returns FNR, FPR given a confusion matrix (binary)'''
	return confusion_m[0, 1] / (confusion_m[0, 1] + confusion_m[1, 1]), confusion_m[1, 0] / (confusion_m[0, 0] + confusion_m[1, 0])


def ROC_DET_arrays(scores, trueL):
	'''
	Receives scores, labels and returns three arrays:\n 
	FNR, FPR and TPR
	'''
	thresholds = np.array(scores)
	thresholds.sort()
	thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
	FNR_arr = np.zeros(thresholds.shape[0])
	FPR_arr = np.zeros(thresholds.shape[0])
	TPR_arr = np.zeros(thresholds.shape[0])
	for idx, t in enumerate(thresholds):
		conf_m_temp = compute_confusion_matrix(compute_optBayes_decisions(scores, given_threshold=t), trueL, 2)
		FNR_temp, FPR_temp = FNR_FPR(conf_m_temp)
		FNR_arr[idx] = FNR_temp
		FPR_arr[idx] = FPR_temp
		TPR_arr[idx] = 1 - FNR_temp
	return FNR_arr, FPR_arr, TPR_arr

def DCF_unnormalized_normalized_min_binary(scores, trueL, triplet, actualOnly=False):
	# Un-normalized
	dcf_u = DCF_binary(compute_confusion_matrix(compute_optBayes_decisions(scores, *triplet), trueL, 2), *triplet)
	# Bayesian risk (with dummy system)
	B_dummy = min(triplet[0] * triplet[1], (1 - triplet[0]) * triplet[2])
	# Normalized Detection Cost Function (wrt. to dummy system)
	dcf_norm = dcf_u / B_dummy
	if actualOnly:
		return (0, dcf_norm, 0)
	# Minimum DCF (with score calibration)
	# Create a new object (ndarray), otherwise in-place sorting will mess up the following computations of the confusion matrix

	thresholds = np.array(scores)
	thresholds.sort()
	thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])

	dcf_min = np.inf

	for threshold in thresholds:
		dcf_temp = DCF_binary(compute_confusion_matrix(compute_optBayes_decisions(scores, *triplet, threshold), trueL, 2), *triplet)
		dcf_temp_norm = dcf_temp / B_dummy
		if dcf_temp_norm < dcf_min:
			dcf_min = dcf_temp_norm

	return (dcf_u, dcf_norm, dcf_min)

def DCF_vs_priorLogOdds(effPriorLogOdds, scores, trueL, actualOnly=False):
	'''Returns normalized and min DCF for a given set of effective prior log-odds'''
	dcf_arr = np.zeros(effPriorLogOdds.shape[0])
	dcfmin_arr = np.zeros(effPriorLogOdds.shape[0])
	for idx, p in enumerate(effPriorLogOdds):
		eff_pi = 1 / (1 + np.exp(-p))
		dcfs = DCF_unnormalized_normalized_min_binary(scores, trueL, (eff_pi, 1, 1), actualOnly=actualOnly)
		dcf_arr[idx] = dcfs[1]
		dcfmin_arr[idx] = dcfs[2]
	return dcf_arr, dcfmin_arr

def compute_misclassification_r(confusion_matrix):
	'''Receives a confusion matrix and computes the mis-classification ratio matrix'''
	column_sum = np.sum(confusion_matrix, axis=0)
	return confusion_matrix / column_sum # Exploit broadcasting

def DCF_unnormalized_normalized_multiclass(prior_p, misclsf_ratios, cost_matrix, norm_term):
	'''Takes formatted inputs, reshapes to 1D when necessary inside the function'''
	element_wise_mul = np.multiply(misclsf_ratios, cost_matrix) # Multiply everything element-wise
	inner_sum = np.sum(element_wise_mul, axis=0) # Sum over each row
	# ravel() to cast into 1D arrays such that np.dot returns a scalar
	dcf_u = np.dot(prior_p.ravel(), inner_sum.ravel())
	dcf_norm = dcf_u / norm_term

	return (dcf_u, dcf_norm)