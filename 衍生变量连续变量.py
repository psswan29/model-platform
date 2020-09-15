# 连续变量衍生
def create_var_new_continuous(data, var, var_new, var_min, var_max):
    '''
    data：输入数据集
    var:连续变量名称
    var_new:衍生变量名
    var_min：原始变量下限
    var_max:原始变量上限
    '''

    T_F = {True: 1, False: 0}
    data[var_new] = data[var].apply(lambda x: (x >= var_min)
                                              & (x < var_max)).map(T_F)
    return data