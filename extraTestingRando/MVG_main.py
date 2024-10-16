import utils as u
import feature_utils as f
import MVG
import numpy as np

# for K-fold
CSF_list = [
    (MVG.gaussianCSF_wrapper, 'Full Covariance Gaussian'), 
    (MVG.naiveBayesGaussianCSF, 'Diag Covariance Gaussian'), 
    (MVG.tiedCovarianceGaussianCSF, 'Tied Full-Cov Gaussian'),
    (MVG.tiedNaiveBayesGaussianCSF, 'Tied Diag-Cov Gaussian')
]

PCA_list = [None, 11, 10]

printStatus = True


calibrate = False
saveCalScores = False

def main():
    print("Starting...")

    gauss_flag=False
    zscore=False

    DTR, LTR = u.load('../dataset/Train.txt')
    DTE, LTE = u.load('../dataset/Test.txt')

    # Reduced dataset (less samples) for testing only
    # DTR, LTR = u.reduced_dataset(DTR, LTR, 1000, seed=0)

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    # ---------------------- MVG classifiers ----------------------

    # priorP = u.vcol(np.array([0.5, 0.5]))
    k = 2 # Number of classes

    n = 4 # Single-Fold value
    K = 5 # K-Fold cross-validation K -> Leave-One-Out if equal to D.shape[1] (number of samples)

    D_merged, L_merged, idxTR_merged, idxTE_merged = u.split_db_after_merge(DTR, DTE, LTR, LTE) # Merged split
    idxTrain_s, idxTest_s = u.split_db_n_to_1(DTR, n) # Single-fold split

    # ----------------- Using validation set (single fold or K-fold) ----------------------
  

    for m in PCA_list:
        pca_msg = '(no PCA)' if m is None else f'(PCA m = {m})'
        # if m is not None:
        #     DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain_s, idxTest_s)[0][0] # Retrieve single fold train subset
        #     PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
        #     DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PCA

        # print('\nSingle Fold ({}-to-1) MVG classifiers {}'.format(n, pca_msg))
        # print('****************************************************')
        # for classifier in CSF_list:
        #     classifier[0](DTR if m is None else DTR_PCA, LTR, k, idxTrain_s, idxTest_s, application_points, show=True, zscore=zscore, gauss_flag=gauss_flag)
        #     print('-----------------------------------------------------')

        # # K-fold
        # scores = MVG.K_fold_MVG(DTR, LTR, k, K, CSF_list, application_points, m, calibrate=calibrate, printStatus=printStatus, returnScores=True if saveCalScores else False,  zscore=zscore, gauss_flag=gauss_flag)
        # if saveCalScores:
        #     np.save('../data_npy/scores_MVG_K_fold_PCA_{}_calibrated.npy'.format(m if m is not None else 'None'), scores)


        # ------------------ Using whole Train.txt dataset and classifying Test.txt (last thing to do) ----------------
        if m is not None:
            DTR_PCA_fold = u.split_dataset(D_merged, L_merged, idxTR_merged, idxTE_merged)[0][0]
            PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over training subset
            D_merged_PCA = np.dot(PCA_Proj.T, D_merged) # Project both training and validation subsets with the output of the PCA
        print('\nMVG classifiers on whole dataset {}'.format(pca_msg))
        print('****************************************************')
        for classifier in CSF_list:
            classifier[0](D_merged if m is None else D_merged_PCA, L_merged, k, idxTR_merged, idxTE_merged, application_points, show=True,  zscore=zscore, gauss_flag=gauss_flag)
            print('-----------------------------------------------------')
         
if __name__  == '__main__':
    main()
    
