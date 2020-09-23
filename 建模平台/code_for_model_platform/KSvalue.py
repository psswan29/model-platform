# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:31:30 2018

@author: surface
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
#计算Ks
from scipy.stats import ks_2samp
def get_ks(y_pred,y_true):
    #y_pred 目标变量预测值 y_true 目标变量真实值
    get_ks1 = ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    return get_ks1

### 计算KS值
def KS(df, score, target):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all_ = pd.DataFrame({'total':total, 'bad':bad})
    all_['good'] = all_['total'] - all_['bad']
    all_[score] = all_.index
    all_ = all_.sort_values(by=score,ascending=False)
    all_.index = range(len(all_))
    all_['badCumRate'] = all_['bad'].cumsum() / all_['bad'].sum()
    all_['goodCumRate'] = all_['good'].cumsum() / all_['good'].sum()
    KS = all_.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return max(KS)

