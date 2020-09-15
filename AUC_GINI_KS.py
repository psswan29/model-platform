#roc曲线和auc值
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import statsmodels.api as sm

def roc_auc_gini(y_true,y_score):
    """
    :param y_true: 目标变量真实值
    :param y_score: 目标变量预测值
    :return:
    roc_auc: AUC值
    gini：基尼系数
    """
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
    print('AUC:',roc_auc,'\n','GINI:',gini,'\n')
    return roc_auc,gini


#计算Ks
from scipy.stats import ks_2samp
def get_ks(y_pred,y_true):
    #y_pred 目标变量预测值 y_true 目标变量真实值
    get_ks1 = ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    return get_ks1