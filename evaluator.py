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



