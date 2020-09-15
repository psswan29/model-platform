import pandas as pd
import numpy as np

# 类别变量统计
def discrete_variable_table(data_x, variable_list):
    """
    此过程为以下其他离散型变量方法的基础
    :param data_x: 数据集
    :param variable_list: 需要统计的变量名列表
    :return:
    discrete_variable_dict:一个以变量名为键，dataframe为值的字典
    """

    discrete_variable_dict = {}
    for k in variable_list:
        kkl = pd.DataFrame(data_x[k].value_counts())  # 创建数据框储存字符变量统计结果
        kkl.rename(columns={k: 'num'}, inplace=True)  # 更改第一列名称计数
        kkl.loc['nan'] = len(data_x[data_x[k].isnull()])  # 添加缺失行统计结果
        kkl['all_data_num'] = len(data_x)  # 添加全部观测数统计结果
        kkl['Proportion'] = kkl['num'] / kkl['all_data_num']  # 添加占比列统计结果
        discrete_variable_dict[k] = kkl
    return discrete_variable_dict


# 数值变量分为图
def fenweishu_continuous_variable(data_1, var_list):
    """
    求连续变量分位数，以此来判断连续变量的同质性
    :param data_1: 数据集
    :param var_list: 需要统计的变量名列表
    :return:
    continuous_variable_dict：一个字典，键为变量名，dataframe为值的字典
    """

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

def discrete_variable_univar(discrete_1):
    """
    离散变量同质分析
    :param discrete_1:
    :return: 一个承装未通过同质性检验变量的列表
    """
    var_tongzhi_list_1 = []
    for i in discrete_1:
        if any(discrete_1[i]['Proportion'] >= 0.9):
            var_tongzhi_list_1.append(i)
            print(i + '\n', discrete_1[i], '\n')
    return var_tongzhi_list_1

def continua_variable_univar(continua_1):
    """
    连续变量同质分析
    :param continua_1:
    :return: 一个承装未通过同质性检验变量的列表
    """
    var_tongzhi_list_2 = []
    for i in continua_1:
        print(i)
        if (continua_1[i]['estimate'].loc[11] >= 0.9) or (
                continua_1[i]['estimate'].loc[2] == continua_1[i]['estimate'].loc[8]):
            var_tongzhi_list_2.append(i)
            print(i + '\n', continua_1[i], '\n')

    return var_tongzhi_list_2