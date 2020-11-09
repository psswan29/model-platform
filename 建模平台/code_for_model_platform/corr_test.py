
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 计算相关系数
from collections import defaultdict

def cor_data(data_1, variable_list):
    '''
    邓欣版本
    data_1:数据集
    variable_list：需要计算相关系数的列表
    '''
    cor_dataframe = data_1[variable_list].corr() # 计算相关系数
    dict_1 = defaultdict(dict)
    for i in cor_dataframe.index:
        for j in cor_dataframe.columns:
            dict_1[i][j] = cor_dataframe.loc[i][j]

    for i in dict_1.keys():
        for j in dict_1[i].keys():
            dict_1[i][j] = round(float(dict_1[i][j]),3)

    # 对值排降序
    for i in dict_1.keys():
        dict_1[i] = sorted(zip(dict_1[i].values(), dict_1[i].keys()), reverse=True)
    return dict_1


def draw_heat(data_1, var_continua_analyse: list):
    """
    绘制相关性热图
    :param data_1: 数据源
    :param var_continua_analyse: 需要分析的连续变量
    :return:
    """
    data_cor_h = data_1[var_continua_analyse].corr()  # test_feature => pandas.DataFrame#
    mask = np.zeros_like(data_cor_h, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(data_cor_h, mask=mask, vmax=1.0, center=0.5,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    return

def tst_continu_var(data_1, var_continua_analyse, corr_rate=0.75):
    """
    :param data_1: 数据源
    :param var_continua_analyse: 需要分析的连续变量, 一个列表
    :param corr_rate: 阈值,用于限定相关性，默认0.75
    :return:
    """
    var_cor = cor_data(data_1, var_continua_analyse)
    var_cor_dict = {}
    # 显示存在相关系数大于0.75的变量
    for i in var_cor:
        if np.abs(var_cor[i][1][0]) >= corr_rate:
            print(i + '\n', var_cor[i][:5], '\n')
            var_cor_dict[i] = var_cor[i]
    return var_cor_dict

# 相关性检验测试版本，基于此版才能进行
def tst_continu_var_1(data_1, var_continua_analyse, corr_rate=0.75):
    """
    created by SHAOMING, WANG
    :param data_1: 数据源
    :param var_continua_analyse:需要分析的连续变量, 一个列表
    :param corr_rate: 阈值,用于限定相关性，默认0.75
    :return:
    corr_dict：相关性字典
    corr_：未通过相关性检验的变量
    independent_var： 通过相关性检验的变量
    log：日志，历史记录
    """
    # 对连续变量进行corr检测，设定大于corr_rate 的两个变量线性相关
    # 得到一个corr_dict字典，它记载每个变量都与哪些其他变量线性相关

    def partitionalise(var_name, corr_dict, corr_, independent_var, log):
        # 若在日志当中，直接跳过
        # 否则加载在日志当中
        if var_name in log: return
        log += list(corr_dict[var_name])

        if not corr_dict[var_name]:
            independent_var.append(var_name)
        else:
            corr_[var_name].add(var_name)
            corr_[var_name].update(corr_dict[var_name])
            for col in corr_dict[var_name]:
                partitionalise(col, corr_dict, corr_, independent_var, log)

    corr_dict = generate_corr_dict(data_1,var_continua_analyse, corr_rate)

    # 挑选出与其他变量线性无关的变量
    # 相当于初始化
    independent_var = []
    from collections import defaultdict
    corr_ =defaultdict(set)
    log = []

    for i in corr_dict.keys():
        partitionalise(i, corr_dict, corr_, independent_var,log)
    return corr_dict, corr_, independent_var,log


def generate_corr_dict(data_1,var_continua_analyse, corr_rate=0.75):
    """
    生成相关性检验字典
    :param data_1:  数据源
    :param var_continua_analyse: 需要分析的连续变量, 一个列表
    :param corr_rate: 阈值,用于限定相关性，默认0.75
    :return: 一个记录相关性检验结果的字典
    """
    c = data_1[var_continua_analyse].corr()

    c = np.abs(c)

    c1 = (c > corr_rate)

    corr_dict = {}

    for col in c1.columns:
        corr_dict[col] = [item for item in c1.index[c1[col]] if item != col]
    return corr_dict

def dengxin_corr_deal(data_1, var_continua_analyse):
    """
    邓欣版本的连续变量自动处理
    :param data_1: 数据源
    :param var_continua_analyse: 一个列表，需要分析的连续型变量名
    :return:
    一个
    """

    var_cor_75_dict = tst_continu_var(data_1, var_continua_analyse)
    list_remove_1 = []
    list_save = []
    for j in var_cor_75_dict.keys():
        list_save.append(j)
        for j2 in var_cor_75_dict[j]:
            if j2[0] >= 0.75 and (j2[1] not in list_save):
                list_remove_1.append(j2[1])

    for var in list_remove_1:
        print('var %s is deleted because of correlation test' % var)
    return ('连续变量自动处理',list_remove_1)