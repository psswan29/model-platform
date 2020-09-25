# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 13:27:50 2018

@author: surface

edited by SHAOMING WANG ON Mon Aug 10, 2020
"""
import pandas as pd
import numpy as np
import math

def model_10_split(y_pred, y_true):
    def get_stats2(group):
        return{'min':group.min(),
               'max':group.max(),
               'count':group.count(),
               'mean':group.mean(),
               'sum':group.sum()}

    data = pd.DataFrame([y_pred,y_true])

    data_sorted = data.sort_values('predict', ascending=False)
    data_sorted['rank_1'] = range(len(data_sorted))
    grouping = pd.qcut(data_sorted.rank_1, 10, labels=False)

    d_act = dict(data_sorted.Y.groupby(grouping).apply(get_stats2).unstack())
    d_pre = dict(data_sorted.predict.groupby(grouping).apply(get_stats2).unstack())

    total_num = d_act['count']
    actual_bad = d_act['sum']
    actual_bad_per = d_act['mean']
    predict_bad = d_pre['mean']

    tat = {'total_count': total_num,
           'actual_bad': actual_bad,
           'actual_bad_per': actual_bad_per,
           'predict_bad': predict_bad}

    tat_1 = pd.DataFrame(tat).fillna(0)
    print(tat_1)
    return tat_1


def model_10_splitm(model, train, target_n='Y'):
    """
    模型10等分，shaoming version
    :param model: 模型
    :param train: 训练数据
    :param target_n: 目标变量名称
    :return:
    """

    train['y_pred_'] = model.predict(train)
    train.sort_values(by='y_pred_', ascending=False,inplace=True)
    train.reset_index(drop=True, inplace=True)

    train['G'] = np.ravel([[i] * int(math.ceil(train.shape[0]/10))  for i in range(10)])

    train_g = train.groupby(by='G').agg({'G':'count',
                                         target_n:['sum','mean'],
                                         'y_pred_':'mean'})
    train_g.columns = ['total_count', 'actual_target', 'actual_p', 'predict_probability']
    return train_g