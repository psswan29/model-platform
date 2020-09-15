#衍生类别变量
def create_var_new_discrete(data,var,var_new,value):
    '''
    data：输入数据集
    var:类别变量
    var_new:衍生变量名
    value:赋值为1的值域,为一个列表
    '''
    T_F = {True:1,False:0}
    data[var_new] = data[var].apply(lambda x:x in value).map(T_F)
    return data

