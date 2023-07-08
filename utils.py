import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy.linalg
import sklearn.datasets
import scipy.optimize as opt
from prettytable import PrettyTable


def mcol(oneDarray):
    return oneDarray.reshape((oneDarray.size, 1))

def mrow(oneDarray):
    return oneDarray.reshape((1, oneDarray.size))

def load(filename):
    DList = []
    labelsList = []

    with open(filename) as f:
        try:
            for line in f:
                attrs = line.replace(" ", "").split(',')[0:12]
                attrs = mcol(np.array([float(i) for i in attrs]))
                _label = line.split(',')[-1].strip()
                label = int(_label)
                DList.append(attrs)
                labelsList.append(label)
        except:
            pass
    return np.hstack(DList), np.array(labelsList, dtype=np.int32)

#shuffle data
def randomize(D, L, seed=0):
    nTrain = int(D.shape[1])
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    
    DTR = D[:, idxTrain]
    LTR = L[idxTrain]
    
    return DTR, LTR
##########################################################################
def Z_normalization(D, means=None, std_v=None):
    '''Returns Z_D (Z-transformed dataset) and the corresponding Mean and Standard Deviation vectors. 
    If applying to a evaluation/validation set, pass the vectors of the corresponding training set obtained before'''
    if means is None and std_v is None:
        mean_vector = mcol(np.mean(D, axis=1))
        std_vector = mcol(np.std(D, axis=1))
        Z_D = (D - mean_vector) / std_vector
        return Z_D, mean_vector, std_vector
    else:
        return (D - means) / std_v
    
def Gaussianization(DTR, DTR_copy):
    P = []
    for dIdx in range(DTR.shape[0]):
        DT = mcol(DTR_copy[dIdx, :])
        X = DTR[dIdx, :] < DT
        print(X)
        R = (X.sum(1) + 1) / (DTR.shape[1] + 2)
        P.append(scipy.stats.norm.ppf(R))
    return np.vstack(P)

#################################################################
def mean(D):
    return mcol(D.mean(1))

def covMatrix(D):
    N = D.shape[1]
    mu_v = mcol(D.mean(axis=1)) 
    centeredD = D - mu_v
    return np.dot(centeredD, centeredD.T) / N

################################################################

def PCA(D, m):
    C = covMatrix(D)

    S, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, :m]

    return P

def LDA(D, L, m, k=2):      #D is the dataset, m is the final desired value of dimension (number of final records), k is number of classes
    n = np.shape(D)[1]   #compute the dataset mean over columns (axis=1) of dataset matrix D
    mu = D.mean(axis=1)

    #Sb
    Sb=0.0              #Sb= 1/n * Σ n_c*((mu_c-mu)*(mu_c-mu).T)     where n_c is number of samples inside class c, mu_c is mean inner to class C
    for i in range(k):
        class_c= D[:, L==i]
        nc_c= class_c.shape[1] #how many columns (records) I have in class i?  
        mu_c=class_c.mean(axis=1)
        mu_c= mu_c.reshape((mu_c.size, 1))
       
        Sb = Sb + (nc_c * np.dot((mu_c-mu), (mu_c-mu).T) )
  
    Sb= Sb / n
    
    #Sw
    Sw = 0
    for i in range(k):      #Sw= 1/n*(n_c * C_c)        where n_c is number of samples inside class c, C_c is covariance matrix inner to class C
        Sw += (L == i).sum() * covMatrix(D[:, L == i], mean(D))

    Sw = Sw / n

    #find W (exatly as P for PCA) direction with eigenvectors
    # associated to maximum eigenvalues of (Sw^-1) * (Sb)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    
    
    return W
##########################################################################
#!log likelihood ratios










#########################################################################
def SW_compute(D, L, k):
    N = D.shape[1] # (number of total samples)

    SW = 0
    for i in range(k):
        # Use samples of each different (c)lass
        Dc = D[:, L == i]
        nc = Dc.shape[1] # n of samples per class
        # Covariance matrix of each class
        Sw = covMatrix(Dc)

        SW += (Sw * nc)

    SW /= N

    return SW


def split_db_n_to_1(D, n, seed=0):
    '''Returns idxTrain and idxTest according to n-to-1 splitting (n is given by the user)'''
    nTrain = int(D.shape[1] * float(n) / float(n + 1))
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) # take a random order of indexes from 0 to N
    idxTrain = idx[0:nTrain] 
    idxTest = idx[nTrain:]
    return idxTrain, idxTest

def split_dataset(D, L, idxTrain, idxTest):
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def split_db_after_merge(DTR, DTE, LTR, LTE):
    '''Returns merged dataset (as if Train and Test data were a single dataset) and the corresponding indexes'''
    D_merged = np.hstack((DTR, DTE))
    L_merged = np.concatenate((LTR, LTE))
    idxTrain = np.arange(0, DTR.shape[1])
    idxTest = np.arange(DTR.shape[1], DTR.shape[1] + DTE.shape[1])
    return D_merged, L_merged, idxTrain, idxTest 

def reduced_dataset(D, L, N, seed=0):
    '''For test purposes. Receives a dataset, its labels and an integer number N. 
    Returns a reduced dataset and labels (of N samples) randomly sampled from the given one'''
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) # take a random order of indexes from 0 to N
    idx_trunc = idx[:N]
    return D[:, idx_trunc], L[idx_trunc]