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