# 交叉列联表
import pandas as pd
import numpy as np


def table_XXX(ddd_1, d1, d2):
    '''
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

# todo
def x_table():
    pass
