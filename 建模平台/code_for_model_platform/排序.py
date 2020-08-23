#排序
def sorttt(data1,var_list,func_list):
    '''
    data1:数据集
    var_list:需要排序的变量 如根据3个['x1','x2','x3']
    func_list:排序的方法 如升降升对应为[True,False,True]
    '''
    data2 = data1.sort_values(var_list,ascending =func_list)
    return data2