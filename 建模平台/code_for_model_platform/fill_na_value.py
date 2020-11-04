#缺失值填补
def missing_tb(data_1, var: str, fill='None'):
    '''
    参数说明：
    data_1: 需处理的数据集
    var:    需处理的变量
    fill:   填补的方式（8 种）：
                None：    不填补
                Drop：    删除
                Mean：    均值填补
                Median：  中位数填补
                Mode：    众数填补
                Max：     最大值填补
                Min：     最小值填补
                '特殊值'：特殊值填补
    '''
        # 填补方式：
    if fill == 'None':
        #  不处理
        print('变量：',var,'没有进行处理')
        data_2 = data_1
    elif fill == 'Drop':
        # 删除
        print('变量：',var,'缺失值被删除')
        data_2 = data_1.dropna(subset=[var])
    elif fill == 'Mean':
        # 均值填补
        print('变量：',var,'用均值',round(data_1[var].mean()), '填补')
        data_2 = data_1.fillna({var:round(data_1[var].mean())})
    elif fill == 'Median':
        # 中位数填补
        print('变量：',var,' 用中位数',data_1[var].median(), '填补')
        data_2 = data_1.fillna({var:data_1[var].median()})
    elif fill == 'Mode':
        # 众数填补
        print('变量：',var,' 用众数',data_1[var].mode()[0], '填补')
        data_2 = data_1.fillna({var:data_1[var].mode()[0]})
    elif fill == 'Max':
        # 最大值填补
        print('变量：',var,' 用最大值',data_1[var].max(), '填补')
        data_2 = data_1.fillna({var:data_1[var].max()})
    elif fill == 'Min':
        # 最小值填补
        print('变量：',var,' 用最小值',data_1[var].min(), '填补')
        data_2 = data_1.fillna({var:data_1[var].min()})
    else:
        # 特殊值填补
        #print('变量：',var,' 用特殊值',fill, '填补')
        data_2 = data_1.fillna({var:fill})
    return data_2

def na_process(data_1, var_discrete, var_continual,fill_value_discrete='nan', fill_value_continual=0):
    """
    缺失值填补，这里使用了一个循环进行缺失值的批量处理
    :param data_1:  源数据
    :param var_discrete:  一个列表，包含所有类别变量
    :param var_continual:  一个列表， 包含所有连续性变量
    :param fill_value_discrete: 类别变量缺失所填的值
    :param fill_value_continual: 连续变量缺失所填的值默认为0，可以是后项值（'bfill','backfill'），
                                前向值（'ffill'）
    :return:
    一份处理后的dataframe
    """
    data_copy = data_1.copy()
    # 遍历所有变量名
    for col in data_copy.columns:

        # 如果变量类型为离散型，则填补缺失值为‘nan’
        if var_discrete and  col in var_discrete:
            data_copy[col].fillna(fill_value_discrete, inplace=True)

        # 否则为连续型，直接补零
        elif var_continual and col in var_continual:

            # 强制转换为数值型
            data_copy[col] = data_copy[col].astype(float)
            # 后项值
            if fill_value_continual in ['bfill','backfill']:
                data_copy[col].backfill(axis=None, inplace=True, limit=None, downcast=None)
            #前向值
            elif fill_value_continual == 'ffill':
                data_copy[col].ffill(axis=None, inplace=True, limit=None, downcast=None)
            else:
                data_copy[col].fillna(fill_value_continual, inplace=True)

    return data_copy
