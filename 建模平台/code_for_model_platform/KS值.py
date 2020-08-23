# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:31:30 2018

@author: surface
"""

from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt  
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import statsmodels.api as sm
#计算Ks
from scipy.stats import ks_2samp
def get_ks(y_pred,y_true):
    #y_pred 目标变量预测值 y_true 目标变量真实值
    get_ks1 = ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    return get_ks1



