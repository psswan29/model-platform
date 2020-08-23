# -*- coding: utf-8 -*-
"""
__________________________________________________________________________
version information:
Created on Tue Apr  3 09:56:19 2018
@author: surface

edited by : Shaoming Wang
on Fri Aug  7 09:56:19 2020
-------------------------------------------------------------------------
logit
    Create a Model from a formula and dataframe.
    
    Parameters
    ----------
    formula : str or generic Formula object
        The formula specifying the model
    data : array-like
        The data for the model. See Notes.
    subset : array-like
        An array-like object of booleans, integers, or index values that
        indicate the subset of df to use in the model. Assumes df is a
        `pandas.DataFrame`
    drop_cols : array-like
        Columns to drop from the design matrix.  Cannot be used to
        drop terms involving categoricals.
    args : extra arguments
        These are passed to the model
    kwargs : extra keyword arguments
        These are passed to the model with one exception. The
        ``eval_env`` keyword is passed to patsy. It can be either a
        :class:`patsy:patsy.EvalEnvironment` object or an integer
        indicating the depth of the namespace to use. For example, the
        default ``eval_env=0`` uses the calling namespace. If you wish
        to use a "clean" environment set ``eval_env=-1``.
    
    Returns
    -------
    model : Model instance
    
    Notes
    -----
    data must define __getitem__ with the keys in the formula terms
    args and kwargs are passed on to the model instantiation. E.g.,
    a numpy structured or rec array, a dictionary, or a pandas DataFrame.
"""

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from icecream import ic
#对样本加权重
def sample_weight(Train_t,y,i_0,j_1):
    """    y:目标变量名称
    i_0：非目标观测权重
    j_1:目标观测权重"""


    i = 0
    T_0 = Train_t[Train_t[y]==0]
    T_1 = Train_t[Train_t[y]==1]
    Train_0_0 = pd.DataFrame()
    Train_1_1 = pd.DataFrame()
    while i < i_0:
        Train_0_0 = pd.concat([Train_0_0,T_0])
        i += 1
    j = 0
    while j < j_1:
        Train_1_1 = pd.concat([Train_1_1,T_1])
        j += 1
    Train_weight = pd.concat([Train_0_0,Train_1_1])
    return Train_weight


def build_logistic_model(y_n, excluded, train):
    model = smf.logit(y_n + " ~ " + "+".join(excluded), train)
    result = model.fit()
    return result

#构造逐步回归筛选变量并建模
def stepwise_selection(train: pd.DataFrame,
                       y_n: str ='Y',
                       initial_list=[],
                       sle=0.05,
                       sls=0.05,
                       verbose=False):
    """
        train - pandas.DataFrame with candidate features 用于训练模型的数据包括因变量
        y - dependent variate 因变量名称，字符型
        initial_list - list of features to start with (column names of X)
        sle - 设定阈值，参数决定新变量是否进入模型
        sls - 设定阈值，参数决定输入变量是否被删除
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features

    Always set threshold_in < threshold_out to avoid infinite looping.

    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """

    X = train[[col for col in train.columns if col != y_n]]
    included = set(initial_list)
    cols = set(X.columns)
    iter_num = 1

    while True:
        print('this is the {} time iteration'.format(iter_num))
        iter_num += 1
        changed = False
        # forward step
        excluded = cols - included
        new_chival = {}

        result  = build_logistic_model(y_n, excluded, train)
        result_w = result.wald_test_terms()

        for new_column in excluded:
            new_chival[new_column] = result_w.summary_frame()['P>chi2'][new_column]
        best_feature, best_chival = sorted(new_chival.items(),
                                          key=lambda x: new_chival[x[0]],
                                          reverse=False)[0]
        ic(best_chival, best_feature)
        
        if best_chival < sle:
            included.add(best_feature)
            ic(included)
            changed=True
            if verbose:
                print('Add  {:30} with chi-square: {:.6}'.format(best_feature, best_chival))

        # backward step
        if len(included) < 3:
            continue
        result_backward = build_logistic_model(y_n, included, train)
        result_backward_w = result_backward.wald_test_terms()
       # use all coefs except intercept
        chivalues = result_backward_w.summary_frame()['P>chi2'].iloc[1:]
        worst_chival = chivalues.max()

        if worst_chival > sls:
            worst_feature = chivalues.index[chivalues.argmax()]
            included.discard(worst_feature)
            if best_feature != worst_feature:
                changed = True
            else:
                changed = False
            if verbose:
                print('Drop {:30} with chi-square: {:.6}'.format(worst_feature, worst_chival))
        if not changed:
            break
    result = build_logistic_model(y_n, included, train)

    return result  #model_t为模型,included为解释变量

def backward_selection(train: pd.DataFrame,
                       y_n: str ='Y',
                       sls=0.05,
                       verbose=False):
    excluded = [col for col in train.columns if col != y_n]
    log = []

    def backward_sub(train, features, y_n, log):
        result = build_logistic_model(y_n, features, train)
        result_w = result.wald_test_terms()
        result_t = result_w.summary_frame()
        feature_select = set(f for f in result_t[result_t['P>chi2'] < sls].index if f != 'Intercept')
        if log and feature_select == log[-1]:
            return result
        else:
            log.append(feature_select)
            return backward_sub(train, feature_select, y_n, log)

    return backward_sub(train, excluded, y_n, log)

if __name__ == '__main__':
    pass


