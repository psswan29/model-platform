#最优变量转换
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics

def var_change(in_df ,X ,Y ,X_min=None,X_max=None):
    """
    :param in_df: 输入数据集
    :param X: 待转换变量
    :param Y: 目标变量
    :param X_min: 待转换变量最小值处理
    :param X_max: 待转换变量最大值处理
    :return:

    """
    in_df[X][in_df[X] >= X_max] = X_max
    in_df[X][in_df[X] <= X_min] = X_min
# 构建中间DataFrame，只保留目标变量"Y"和待转换变量"X"，并进行相应转换
    df_1 = in_df[in_df[X].notnull()]
    df_1["X_sq"] = df_1[X] ** 2
    df_1["X_sqrt"] = np.sqrt([x if x>0 else 0 for x in df_1[X]])  # sqrt(max(x,0))
    df_1["X_cu"] = df_1[X] ** 3
    df_1["X_curt"] = np.cbrt(df_1[X])
    df_1["X_log"] = np.log([x if x>0.001 else 0.001 for x in df_1[X]])  # log(max(x,0.001))
    df_1["X_re"] = [1/x if x>0.001 else 1/0.001 for x in df_1[X]]  # 1/(max(x,0.001))
# 从7种转换方法种，挑选最优的转换方法。用"statsmodels"下的Logit，逐一与y进行建模，并计算每个转化后变量的p_value。
    column=[X,"X_sq","X_sqrt","X_cu","X_curt","X_log","X_re"]
    var_auc = pd.Series()
    for new_column in column:
        model = sm.Logit(df_1[Y], sm.add_constant(df_1[new_column])).fit()
        df_1['predict'] = model.predict(sm.add_constant(df_1[new_column]))
        var_auc[new_column] = metrics.roc_auc_score(df_1[Y],df_1['predict'])
    max_tran = var_auc.argmax()
    best_tr = {'tr_func':max_tran,'AUC':var_auc[max_tran]}
    var_auc = var_auc.sort_values(ascending=False)
    return best_tr, var_auc