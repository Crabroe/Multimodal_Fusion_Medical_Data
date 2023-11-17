import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,auc,confusion_matrix

def _roc(this_test_y, pred):
    roc=roc_auc_score(this_test_y, pred)
    return roc

def _sensitivity(this_test_y, pred):
    sensitivity=confusion_matrix(this_test_y, pred)[0][0]/(confusion_matrix(this_test_y, pred)[0][0]+confusion_matrix(this_test_y, pred)[0][1])
    return sensitivity

def _specify(this_test_y, pred):
    specificity=confusion_matrix(this_test_y, pred)[1][1]/(confusion_matrix(this_test_y, pred)[1][1]+confusion_matrix(this_test_y, pred)[1][0])
    return specificity

def mat_roc(mat: np.ndarray):
    this_test_y = []
    pred = []
    for i in range(2):
        for j in range(2):
            for n in range(mat[i, j]):
                this_test_y.append(i)
                pred.append(j)
    return _roc(this_test_y, pred)

def mat_sensitivity(mat: np.ndarray):
    sensitivity = mat[0][0] / (mat[0][0] + mat[0][1])
    return sensitivity

def mat_specify(mat: np.ndarray):
    specificity = mat[1][1] / (mat[1][1] + mat[1][0])
    return specificity