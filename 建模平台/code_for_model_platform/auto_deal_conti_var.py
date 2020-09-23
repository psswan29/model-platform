
def auto_deal_continua(var_continua_analyse_2, data_1):
    """
    # 连续变量自动处理
    :param var_continua_analyse_2:
    :param data_1:
    :return:
    var_continua_for_model: 用于建模的连续变量
     var_continua_process: 变量处理过程
    """
    from .auto_bin_using_chi import moto_binning


    var_continua_for_model = []
    var_continua_process = {}
    for j1 in var_continua_analyse_2:
        auto_dict = moto_binning(data_1, j1, 'Y')
        new_col = j1 + '_1'
        data_1[new_col] = auto_dict[0]
        var_continua_process[j1] = auto_dict[1]
        var_continua_for_model.append(new_col)
        # print(var_continua_process[j1])

    # 结果显示连续变量自动处理结果
    for i in var_continua_process.items():
        print(i, '\n')
    return var_continua_for_model, var_continua_process