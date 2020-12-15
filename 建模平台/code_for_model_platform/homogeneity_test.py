import pandas as pd
import numpy as np

# 类别变量统计
def discrete_variable_table(data_1, variable_list):
    '''
    返回一个离散型变量字典
    data_1:数据集
    variable_list：需要统计的变量名列表
    '''

    discrete_variable_dict = {}
    for k in variable_list:
        kkl = pd.DataFrame(data_1[k].value_counts())  # 创建数据框储存字符变量统计结果
        kkl.rename(columns={k: 'num'}, inplace=True)  # 更改第一列名称计数
        kkl.loc['nan'] = len(data_1[data_1[k].isnull()])  # 添加缺失行统计结果
        kkl['all_data_num'] = len(data_1)  # 添加全部观测数统计结果
        kkl['Proportion'] = kkl['num'] / kkl['all_data_num']  # 添加占比列统计结果
        discrete_variable_dict[k] = kkl
    return discrete_variable_dict


# 数值变量分为图
def fenweishu_continuous_variable(data_1, var_list):
    '''
    data_1:数据集
    var_list：需要统计的变量名列表
    '''
    continuous_variable_dict = {}
    for i in var_list:
        fenw = pd.DataFrame(columns=['percentile', 'estimate'])
        fenw['percentile'] = ['100%max', '99%', '90%', '75% Q3', '50% Median',
                              '25% Q1', '10%', '5%', '1%', '0% Min',
                              'Nan', 'Nan_rate']
        n_1 = [100, 99, 90, 75, 50, 25, 10, 5, 1, 0]
        x_1 = [np.percentile(data_1[i].dropna(), x) for x in n_1]
        x_1.append(data_1[i].isnull().sum())
        x_1.append(data_1[i].isnull().sum() / len(data_1))
        fenw['estimate'] = x_1
        continuous_variable_dict[i] = fenw
    return continuous_variable_dict

def discrete_variable_univar(discrete_1, p=0.9, verbose=False):
    """
    离散变量同质分析
    :param discrete_1: 一个字典，键是变量名，值是series
    :return: 一个列表
    """
    var_tongzhi_list_1 = []
    for i in discrete_1:
        if any(discrete_1[i]['Proportion'] >= p):
            var_tongzhi_list_1.append(i)
            if verbose:
                print(i + '\n', discrete_1[i], '\n')
    return ('离散变量同质分析', var_tongzhi_list_1)

def continua_variable_univar(continua_1, verbose=False):
    """
    连续变量同质分析，邓欣版本
    :param continua_1: 一个字典：键是变量名，值是series
    :return: 一个列表
    """
    var_tongzhi_list_2 = []
    for i in continua_1:
        if verbose:
            print('同质分析:',i)
        if (continua_1[i]['estimate'].loc[11] >= 0.9) or (
                continua_1[i]['estimate'].loc[2] == continua_1[i]['estimate'].loc[8]):
            var_tongzhi_list_2.append(i)
            if verbose:
                print(i + '\n', continua_1[i], '\n')

    return ('连续变量同质分析',var_tongzhi_list_2)

def homogeneity_test_m(continua_var, p=3,step=5):
    """
    连续变量同质分析，王绍明版本
    :param continua_1: 一个连续变量pandas.series
    :param p:反应了多少个分位点
    :return: boolean value, 是否通过同质性检验
    """
    # 产生21个分位点，去重之后
    if len(set(continua_var.quantile([i * 0.01  for i in range(0,101,step)],'nearest'))) <= p:
        return False
    else:
        return True

def continua_variable_univar_m(dataset, continual_vars):
    """
    连续变量同质性分析，王绍明版本

    :param dataset: 原数据集
    :param continual_vars: 需要进行检验的连续变量
    :return: 一个列表
    """
    tongzhi_vars = []
    for col in continual_vars:
        if not homogeneity_test_m(dataset[col]):
            tongzhi_vars.append(col)

    return ('连续变量同质性分析',tongzhi_vars)









