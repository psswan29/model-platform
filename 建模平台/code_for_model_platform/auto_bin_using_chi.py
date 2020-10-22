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

def auto_deal_continua(var_continua_analyse_2, data_1):
    """
    # 连续变量自动处理
    :param var_continua_analyse_2:
    :param data_1:
    :return:
    var_continua_for_model: 用于建模的连续变量
     var_continua_process: 变量处理过程
    """


    var_continua_for_model = []
    var_continua_process = {}
    for j1 in var_continua_analyse_2:
        auto_dict = moto_binning_chi(data_1, j1, 'Y')
        new_col = j1 + '_1'
        data_1[new_col] = auto_dict[0]
        var_continua_process[j1] = auto_dict[1]
        var_continua_for_model.append(new_col)
        # print(var_continua_process[j1])

    # 结果显示连续变量自动处理结果
    for i in var_continua_process.items():
        print(i, '\n')
    return var_continua_for_model, var_continua_process

def moto_binning_chi(data_0, var_continu: str, y:str):
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
            new_var_dict['>%s and <=%s'%(i_1, j_1)] = ((data_1['%s'%var_continu] > i_1) & (data_1['%s'%var_continu] <= j_1)).astype(int)
    output_2.sort_values(ascending=False, inplace=True)
    try:
        output_2_1 = output_2.iloc[:5]
    except:
        output_2_1 = output_2

    output_final = pd.concat([output_1_1, output_2_1])
    ix = output_final.index[output_final.argmax()]
    new_var_final = new_var_dict[ix]
    return new_var_final, ix

def bin_using_ks(var_continuous, y, threshold=0.05):
    """
    使用best——ks方法实现自动分箱
    :param var_continuous:
    :param y:
    :param threshold: 阈值，
    :return:
    """


    def bin_using_ks_(var_len, var_continuous, y, log, threshold):
        """
        使用递归的方法分箱内部方法
        :param var_len: 要分箱的连续变量初始长度
        :param var_continu: 每次迭代要分箱的连续变量数据
        :param y: 每次迭代因变量数据
        :return:

        递归终止条件

        1.下一步分箱后,最小的箱的占比低于设定的阈值(threshold常用0.05)
        2.下一步分箱后,该箱对应的y类别全部为0或者1
        3.下一步分箱后,因变量反馈率不单调

        缺点
        不能用于多分类问题

        """


        # 对输入变量进行处理，然后进行检验，保证
        var_continuous = pd.DataFrame(var_continuous).values.reshape(-1,)
        y = pd.DataFrame(y).values.reshape(-1,)
        assert var_continuous.shape == y.shape
        # 正负样本比例检验
        if ((y == 1).sum()/var_len < threshold) or ((y ==0).sum()/var_len < threshold):
            raise ImportError

        value_set= set(var_continuous)

        ks_list = [(i,get_ks((var_continuous<=i).astype(int), y))    for i in value_set]
        value, ks = sorted(ks_list,key=lambda x: x[1],reverse=True)[0]
        log.append(value)

        scalar1 = (var_continuous < value)
        scalar2 = (var_continuous >= value)

        # 下一步分箱后,最小的箱的占比低于设定的阈值(threshold常用0.05)
        if (scalar1.sum()/var_len < threshold) or (scalar2.sum()/var_len < threshold):
            return log
        # 下一步分箱后, 该箱对应的y类别全部为0或者1
        if len(set(y[scalar1]))==1 or len(set(y[scalar2]))==1:
            return log
        # 下一步分箱后,bad rate 不单调
        if monotonicity_test(var_continuous,y, log):
            return log

        return bin_using_ks(var_len, var_continuous[scalar1], y[scalar1], log, threshold), \
               bin_using_ks(var_len, var_continuous[scalar2], y[scalar2], log, threshold)

    # 反馈率单调性检验
    def monotonicity_test(var_continuous,y, log):
        values = []
        log = sorted(log)
        state = set()
        for i,e in enumerate(log[:-1]):
            if i == 0:
                values.append(y[(var_continuous < log[i])].mean())
                continue
            values.append(y[(var_continuous >= log[i]) & (var_continuous < log[i+1])].mean())
            if i == (len(log)-2):
                values.append(y[(var_continuous >= log[i])].mean())
        for i,v in enumerate(values[:-1]):
            if values[i]>values[i+1]:
                state.update(0)
            else:
                state.update(1)
            if len(state)>1:
                return False
        return  True


    log = []
    var_len = var_continuous.shape[0]
    bin_using_ks_(var_len, var_continuous, y, log, threshold)

    return log


from scipy.stats import ks_2samp
def get_ks(y_pred,y_true):
    #y_pred 目标变量预测值 y_true 目标变量真实值
    get_ks1 = ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    return get_ks1

def chi_merge_base(data_0, var_continu: str, y:str):
    # 基于卡方检验,最原始的卡方分箱法
    # 首先定义一个阈值 threshold
    # 第一步，初始化，根据要离散的变量，对实例进行排序：每个实例属于一个区间
    # 第二步，合并区间
    # 计算每一对之间的chi square value
    # 将chi square value 最小的两个区间合并

    pass


if __name__ == '__main__':
    df = pd.read_csv('./lucheng_data.csv', encoding='gbk')
    output = moto_binning_chi(df, 'WEIXIN_APP_NUM_M1', 'Y')
    output = bin_using_ks()