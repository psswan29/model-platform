#基尼系数
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def roc_auc_gini(y_score,y_true):
    '''
    y_true:目标变量真实值
    y_score:目标变量预测值
    '''
    fpr, tpr, thresholds = roc_curve(y_true,y_score,pos_label = 1)
    plt.plot(fpr,tpr,linewidth=2,label="ROC")
    plt.xlabel("false presitive rate")
    plt.ylabel("true presitive rate")
    plt.ylim(0,1.05)
    plt.xlim(0,1.05)
    plt.legend(loc=4)
    plt.show()
    roc_auc = auc(fpr, tpr)
    gini = 2*roc_auc - 1
    return gini