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
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from scipy import stats

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


def build_logistic_model(y_name, X_names, train,seed=2020,maxiter=None):
    """
    建立逻辑回归模型，此方法是使用stats的内嵌方法，不具备变量筛选功能，
    stepwise_selection以及backward_selection是以此方法为基础
    :param y_name: 因变量名称
    :param X_names:解释变量名称
    :param train: 训练集
    :return:
    result 建模结果
    """
    random.seed(seed)
    np.random.seed(seed)

    model = smf.logit(y_name + " ~ " + "+".join(X_names), train)
    if not maxiter:
        maxiter = train.shape[0]
    result = model.fit(maxiter=maxiter)
    return result

def build_logistic_model_skl(X_train, y_train, **kwargs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import chi2

    np.random.seed(2020)
    import random
    random.seed(2020)

    LG = LogisticRegression(penalty='l2',
                            dual=False,
                            tol=0.1e-5,
                            C=1.0,
                            fit_intercept=True,
                            intercept_scaling=1,
                            class_weight=None,
                            random_state=2020,
                            solver='newton-cg',
                            max_iter=1000,
                            multi_class='auto',
                            verbose=10,
                            warm_start=False,
                            n_jobs=4,
                            l1_ratio=None)
    LG.fit( X_train,y_train , sample_weight=None)
    print(LG.get_params())
    auc = roc_auc_score(y_train, LG.predict_proba(X_train)[:, 1])
    t= get_tvalues(LG,X_train,y_train)
    return LG, auc, t

def get_tvalues(lm,X,y):
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict_proba(X)[:,1]

    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)

    index = ['intercept_']+list(X.columns)
    myDF3 = pd.DataFrame(index=index)
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["P_value"] = [params,sd_b,ts_b,p_values]
    return myDF3


def stepwise_selection_skl(train: pd.DataFrame,
                       y_n: str ='Y',
                       initial_list=[],
                       sle=0.05,
                       sls=0.05,
                       verbose=False,
                           min_var_num=3):
    """
        created by SHAOMING, WANG
        sklearn 版本逐步回归
        train - pandas.DataFrame with candidate features 用于训练模型的数据需包括因变量
        y - dependent variate 因变量名称，字符型
        initial_list - list of features to start with (column names of X)
        sle - 设定阈值，参数决定新变量是否进入模型
        sls - 设定阈值，参数决定输入变量是否被删除
        verbose - 是否打印进入模型变量以及排除模型变量
    Returns:

    Always set threshold_in < threshold_out to avoid infinite looping.

    """

    X = train[[col for col in train.columns if col != y_n]]
    included = set(initial_list)
    cols = set(X.columns)
    # 迭代次数
    iter_num = 1

    while True:
        print('this is the {} time iteration'.format(iter_num))
        iter_num += 1
        changed = False

        excluded = cols - included
        # new_chival = {}

        model,auc, t = build_logistic_model_skl(X.loc[:,list(excluded)],train[y_n])
        best_feature_array = t.iloc[1:,:].sort_values('P_value').iloc[0, :]
        best_feature, best_chival = best_feature_array.name, best_feature_array['P_value']

        # result  = build_logistic_model(y_n, excluded, train)
        # result_w = result.wald_test_terms()

        # for new_column in excluded:
        #     new_chival[new_column] = t['P_value'][new_column]
        # best_feature, best_chival = sorted(new_chival.items(),
        #                                   key=lambda x: x[1],
        #                                   reverse=False)[0]
        if best_chival < sle:
            included.add(best_feature)
            if  verbose:
                print(included)
                print('Add  {:30} with chi-square p-value: {:.6}'.format(best_feature, best_chival))
            changed=True

        if len(included) < min_var_num:
            continue
        model,auc, t = build_logistic_model_skl(X.loc[:,list(included)],train[y_n])

        worst_feature = t.index[np.argmax(t['P_value'])]
        # result_backward = build_logistic_model(y_n, included, train)
        # result_backward_w = result_backward.wald_test_terms()
        # use all coefs except intercept
        #  pvalues = result_backward_w.summary_frame()['P_value'].iloc[1:]
        worst_chip = t['P_value'].max()
        # worst_feature = worst_chip.index[worst_chip.argmax()]

        if worst_chip > sls:
            included.discard(worst_feature)
            if best_feature != worst_feature:
                changed = True
            else:
                changed = False
            if verbose:
                print('Drop {:30} with chi-square p-value: {:.6}'.format(worst_feature, worst_chip))
        if not changed:
            break

    # result = build_logistic_model(y_n, included, train)
    model, auc, t = build_logistic_model_skl(X.loc[:, list(included)], train[y_n])

    return model, auc, t

def stepwise_selection(train: pd.DataFrame,
                       initial_list:list,
                       y_n: str ='Y',
                       sle=0.05,
                       sls=0.05,
                       verbose=False,
                       min_var_num=3,
                       max_var_num=15,
                       strict_mode=False):
    """
        created by SHAOMING, WANG
        train - pandas.DataFrame with candidate features 用于训练模型的数据需包括因变量
        y - dependent variate 因变量名称，字符型
        initial_list - list of features to start with (column names of X)
        sle - 设定阈值，参数决定新变量是否进入模型
        sls - 设定阈值，参数决定输入变量是否被删除
        verbose - 是否打印进入模型变量以及排除模型变量
        min_var_num - 设置最小入模变量数，默认3
        max_var_num - 设置最大入模变量数，默认·15
        strict_mode - 严格·模式， 关系到入模变量的数目，默认False
    Returns:

    Always set threshold_in < threshold_out to avoid infinite looping.

    """

    func_dict = {'max':(np.max, np.argmax),'min':(np.min,np.argmin)}

    def choice_waldtst(included, f_n):
        result_backward = build_logistic_model(y_n, included, train)
        result_backward_w = result_backward.wald_test_terms()
        f1 = func_dict[f_n][0]
        f2 = func_dict[f_n][1]
        chivalues = result_backward_w.summary_frame()['P>chi2'].iloc[1:]
        p_chival = f1(chivalues)
        feature = chivalues.index[f2(chivalues)]
        return p_chival, feature

    X = train[[col for col in train.columns if col != y_n]]
    included = initial_list
    cols = set(X.columns)
    # 迭代次数
    iter_num = 1
    history = []

    while True:
        print('this is the {} time iteration'.format(iter_num))
        iter_num += 1
        changed = False

        excluded = [col for col in cols if (col not in included) and (col not in history)]

        best_chival, best_feature = choice_waldtst(excluded, 'min')
        history.append(best_feature)

        if best_chival < sle:
            included.append(best_feature)
            if  verbose:
                print('include:',included)
                print('Add  {:30} with chi-square p-value: {:.6}'.format(best_feature, best_chival))
            changed=True

        if len(included) < min_var_num:
            continue

        worst_chival, worst_feature = choice_waldtst(included,'max')

        if worst_chival > sls:
            included.remove(worst_feature)
            if strict_mode:
                if best_feature != worst_feature:
                    changed = True
                else:
                    changed = False
            if verbose:
                print('Drop {:30} with chi-square p-value: {:.6}'.format(worst_feature, worst_chival))

        if (len(included) >= max_var_num) or (not changed):
            break

    result = build_logistic_model(y_n, included, train)

    return result
#构造逐步回归筛选变量并建模

def backward_selection(train: pd.DataFrame,
                       y_name: str ='Y',
                       sls=0.05,
                       verbose=False):
    """
    反向淘汰法, 使用递归的方法求
    :param train: 训练数据
    :param y_n: 因变量名称
    :param sls: 设定阈值，参数决定输入变量是否被删除
    verbose: 是否打印日志
    :return:
    result: 模型结果
    """
    excluded = [col for col in train.columns if col != y_name]
    log = []

    def backward_sub(train, features, y_n, log):
        result = build_logistic_model(y_n, features, train)
        result_w = result.wald_test_terms()
        result_t = result_w.summary_frame()
        feature_select = [f for f in result_t[result_t['P>chi2'] < sls].index if f != 'Intercept']
        if log and feature_select == log[-1]:
            if verbose:
                print('the log of building the model is ', log)
            return result
        else:
            log.append(feature_select)
            return backward_sub(train, feature_select, y_n, log)

    return backward_sub(train, excluded, y_name, log)

if __name__ == '__main__':
    pass


