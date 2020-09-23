# 交叉列联表
import pandas as pd
import numpy as np
from functools import lru_cache

def table_XXX(ddd_1, d1, d2):
    '''
    created by Dengxin
    ddd_3:数据集
    d1:X变量，输入变量名称
    d2:Y变量，输入变量名称
    '''
    ddd_3 = ddd_1.fillna({d1: 'Nan', d2: 'Nan'})  # 把缺失标记为'Nan'

    ddd_3['freq'] = 1

    #
    list33 = pd.DataFrame(pd.pivot_table(ddd_3, index=[d1], columns=[d2], values=['freq'],
                                         aggfunc=[np.sum], fill_value=0))
    list44 = list33.copy()

    list33.loc['总计'] = pd.Series(list33.apply(sum, axis=0))
    list33['总计'] = list33.apply(sum, axis=1)

    list33['type'] = '频次'  # 1

    # 算反馈率
    zz1 = pd.Series(list44.apply(sum, axis=1))
    list44_1 = pd.DataFrame(list44.apply(lambda x: x * 100 / zz1, axis=0))
    list44_1 = list44_1.apply(lambda x: round(x, 2))
    list44_1['type'] = '反馈率'  # 3

    # 算列占比
    p1 = pd.Series((list44.apply(lambda x: x.sum(), axis=0)))
    list44_2 = list44.apply(lambda x: x * 100 / p1, axis=1)
    list44_2 = list44_2.apply(lambda x: round(x, 2))
    list44_2['type'] = '列占比'  # 4

    # 计算占比
    list44_3 = list44.apply(lambda x: x * 100 / len(ddd_3))
    list44_3.loc['总计'] = pd.Series(list44_3.apply(sum, axis=0))
    list44_3['总计'] = list44_3.apply(sum, axis=1)
    list44_3 = list44_3.apply(lambda x: round(x, 2))
    list44_3['type'] = '占比'  # 2

    list_final_1 = pd.concat([list33, list44_1, list44_2, list44_3])
    list_final_1['值'] = list_final_1.index
    tt = pd.pivot_table(list_final_1, index=['值', 'type'])

    # 重排顺序
    kk = list(tt.index)
    kk.remove(('总计', '频次'))
    kk.remove(('总计', '占比'))
    kk_2 = []
    for sf in set(ddd_3[d1]):
        kk_2.append((sf, '频次'))
        kk_2.append((sf, '占比'))
        kk_2.append((sf, '反馈率'))
        kk_2.append((sf, '列占比'))
    kk_2.append(('总计', '频次'))
    kk_2.append(('总计', '占比'))
    tt2 = tt.reindex(kk_2)
    # 改行名
    kk_3 = []
    for js in set(ddd_3[d2]):
        kk_3.append((js))
    kk_3.append('总计')
    kk_4 = [d2] * (len(set(ddd_3[d2])) + 1)
    tt2.columns = [kk_4, kk_3]
    # print(tt2)
    return tt2

#  建立一个英文版的交叉列联表
def x_table(data, X_NAME, Y_NAME):
    """
    建立一个英文版的交叉列联表，解决中文版的乱码问题
    created by Shaoming, Wang
    :param data: 源数据
    :param X_NAME:  输入变量名称
    :param Y_NAME: 因变量名称
    :return:
    """
    data[X_NAME].fillna('nan',inplace=True)
    x_vs = data[X_NAME].unique()
    type = ['Freq', 'Prop', 'Response_rate', 'col_prop']
    t = [[i, k] for i in x_vs for k in type]

    # 计算频率
    def get_freq(value, data, X_NAME, Y_NAME):
        """
        :param value: 特征值是什么
        :param data: 源数据
        :param X_NAME: 变量名或者特征名称
        :return: 第一个是Y等于零，第二个是y等于1，第三个是总计

        """
        t = data[(data[X_NAME] == value)]
        t0 = (t[Y_NAME] == 0).sum()
        t1 = (t[Y_NAME] == 1).sum()
        return  t0,t1, t.shape[0]

    # 计算占比
    def get_prop(value, data, X_NAME, Y_NAME):
        data_shape = data.shape[0]
        t0,t1, total = get_freq(value, data, X_NAME, Y_NAME)
        return t0/data_shape, t1/data_shape, total/data_shape

    # 计算反馈率/行占比
    def get_res_rate(value, data, X_NAME, Y_NAME):
        t0,t1, total = get_freq(value, data, X_NAME, Y_NAME)
        return t0/(total+1), t1/(total+1), total/(total+1)

    # 计算列比率
    def get_col_prop(value, data, X_NAME, Y_NAME):
        num_y0 = (data[Y_NAME] == 0).sum()
        num_y1 = (data[Y_NAME] == 1).sum()

        t0, t1, total = get_freq(value, data, X_NAME, Y_NAME)
        return  (t0+1)/(num_y0+1), (t1+1)/(num_y1+1), None

    dict_ = {'Freq': get_freq,
             'Prop': get_prop,
             'Response_rate': get_res_rate,
             'col_prop': get_col_prop }

    for i, x in enumerate(t):
        t[i] += list(dict_[x[1]](x[0], data, X_NAME, Y_NAME))

    # 生成最后‘总计’行
    num_y0 = (data[Y_NAME] == 0).sum()
    num_y1 = (data[Y_NAME] == 1).sum()
    t.append(['Total','Freq',num_y0, num_y1,data.shape[0]])
    t.append(['Total','Prop',num_y0/data.shape[0], num_y1/data.shape[0],1])

    # 转换为dataframe
    T = pd.DataFrame(t)
    T.columns = [X_NAME,'type', 'Y_0','Y_1','Total']
    T.set_index([X_NAME,'type'],inplace=True)

    print(T)

    return T
