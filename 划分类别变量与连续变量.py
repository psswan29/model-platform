from collections import defaultdict


def divide_cat_con_var(data_X, var_list, num=40, rate=0.2):
    """
    用于划分类别变量以及连续变量，其余归为其他
    :param data_X:  数据源
    :param var_list:  选用的变量列表
    :return: 一个记载分类结果的字典 continua 连续变量 discrete 类别变量 other 其他类型
    """
    var_type_dict = defaultdict(list)
    for i in var_list:
        if (data_X[i].dtypes in [object]):
            var_type_dict['discrete'].append(i)

        elif (data_X[i].dtypes in [float, int]):
            data_1_p = len(set(data_X[i]))
            # 一般
            if data_1_p > num and  data_1_p > data_X.shape[0] * rate:
                var_type_dict['continua'].append(i)
            else:
                var_type_dict['discrete'].append(i)
        # 这里偶尔会出现日期格式
        else:
            var_type_dict['other'].append(i)
    return var_type_dict

def judge_leibie_lianxu(data_X, divide_value_min=40, divide_value_max=80):
    """
    同样用于判断类别变量与连续变量，判定方法依然有待商榷
    在邓欣版本的基础上，添加改动
    :param data_X:
    :param divide_value_min: 默认40 用于判断一个变量所有实例（观测）的种类是否超过这个值
    :param divide_value_max:  默认 80
    :return:
        var_type_dict 一个记载分类结果的字典
        continua 连续变量
        discrete 类别变量
        other 其他类型
    """
    var_type_dict = defaultdict(list)
    for i in data_X:
        if len(set(data_X[i])) > divide_value_min and (data_X[i].dtypes in (float, int)):
            var_type_dict['continua'].append(i)
        elif len(set(data_X[i])) < divide_value_max:
            var_type_dict['discrete'].append(i)
        else:
            var_type_dict['other'].append(i)
    return var_type_dict