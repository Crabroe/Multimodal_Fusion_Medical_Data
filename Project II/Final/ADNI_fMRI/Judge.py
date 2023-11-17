import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, auc, confusion_matrix, roc_curve


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

def auc_sens_spe_cal(real,pred,draw='false'):
    fpr,tpr,threshold = roc_curve(real, pred)
    roc_auc = auc(fpr,tpr)
    if(draw=='true'):
        plt.figure()
        lw = 2
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='test ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
    if(len(set(real))!=1):
        roc=roc_auc_score(real, pred)
        sensitivity=confusion_matrix(real, pred)[0][0]/(confusion_matrix(real, pred)[0][0]+confusion_matrix(real, pred)[0][1])
        specificity=confusion_matrix(real, pred)[1][1]/(confusion_matrix(real, pred)[1][1]+confusion_matrix(real, pred)[1][0])
    else:
        roc=0
        sensitivity=0
        specificity=0
    return roc,sensitivity,specificity