# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:18:22 2018
@author: surface

Edicted by SHAOMINGWANG
ON Tue Aug 26 12:18:22 2020

"""

from scipy.stats import chisquare  
import numpy as np
import pandas as pd
from icecream import ic
from scipy.stats import chi2_contingency
import os

def moto_binning(data_0, var_continu: str, y:str):
    """
    #基于卡方分层
    #
    :param data_0: 数据源
    :param var_continu: 用于分层的连续变量
    :param y: 因变量
    :return:
        new_var_final: new_var_dict 中保存的最优的处理后的vector
        ix: new_var_dict 中键值
    """
    new_var_dict = {}
    data_1 = data_0.copy()
    
    xx = [x*2 for x in range(1, 50)]

    xx_2 = [np.percentile(data_1[var_continu].dropna(), x) for x in xx] #按比例算分位点
    
    output_1 = pd.Series()
    for x in set(xx_2):
        data_1['new_var'] = (data_1[var_continu] <= x).astype(int)
        pp = pd.crosstab(data_1['%s' % y],data_1['new_var'])
        dd = np.array([list(pp.iloc[0]),list(pp.iloc[1])])
        pppp = chi2_contingency(dd)
        output_1['<=%s' % x] = pppp[0]
        new_var_dict['<=%s' % x] = (data_1[var_continu] <= x).astype(int)
    output_1.sort_values(ascending=False, inplace=True)
    try:
        output_1_1 = output_1.iloc[:5]
    except:
        print('error')
        output_1_1 = output_1

    output_2 = pd.Series()
    for i in range(1, 45):
        for j in range(i+5, 50):
            i_1 = np.percentile(data_1[var_continu].dropna(), i*2)
            j_1 = np.percentile(data_1[var_continu].dropna(), j*2)
            data_1['new_var_2'] = ((data_1[var_continu] > i_1) & (data_1[var_continu]<=j_1)).astype(int)
            pp_2 = pd.crosstab(data_1['%s' % y], data_1['new_var_2'])
            dd_2 = np.array([list(pp_2.iloc[0]), list(pp_2.iloc[1])])
            pppp_2 = chi2_contingency(dd_2)
            output_2['>%s and <=%s'%(i_1, j_1)] = pppp_2[0]
            new_var_dict['>%s and <=%s'%(i_1, j_1)
             ] = ((data_1['%s'%var_continu] > i_1) & (data_1['%s'%var_continu] <= j_1)).astype(int)
    output_2.sort_values(ascending=False, inplace=True)
    try:
        output_2_1 = output_2.iloc[:5]
    except:
        output_2_1 = output_2

    output_final = pd.concat([output_1_1, output_2_1])
    ix = output_final.index[output_final.argmax()]
    new_var_final = new_var_dict[ix]
    return new_var_final, ix


if __name__ == '__main__':
    df = pd.read_csv('../lucheng_data.csv', encoding='gbk')
    output = moto_binning(df, 'WEIXIN_APP_NUM_M1', 'Y')
    