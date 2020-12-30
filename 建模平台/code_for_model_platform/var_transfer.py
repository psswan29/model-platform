#最优变量转换
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics


def var_change(in_df:pd.DataFrame,
               X:str ,
               Y:str ,
               X_min=None,
               X_max=None,
               n=20,
               threshold=None):
    """
    :param in_df: 输入数据集
    :param X: 待转换变量
    :param Y: 目标变量
    :param X_min: 待转换变量最小值处理
    :param X_max: 待转换变量最大值处理
    :return:

    """
    if (X_min is None) and (X_max is None):
        df = in_df[[X,Y]].copy()

        min_value = df.iloc[:,0].min()
        max_value = df.iloc[:,0].max()
        range_ = max_value - min_value

        step = np.ceil(range_/n)
        dots = [min_value + i*step for i in range(n+1)]
        edges = [(dots[i], dots[i+1]) for i in range(n)]

        vs = [df[Y][((df[X]>=down)&(df[X]<up))].mean() for down, up in edges]

        if not threshold:
            threshold = df[Y].mean() * 0.05

        vs_mean = np.where(np.isnan(vs), 0, vs)
        temp = (np.abs(np.diff(vs_mean))<threshold)
        hill = [temp[i] ^ temp[i + 1] for i in range(len(temp) - 1)]

        if temp[0]:
            pos_min = np.argmax(hill) + 1
            X_min = dots[pos_min]
        if temp[-1]:
            pos_max = -(np.argmax(hill[::-1])+1)-1
            X_max = dots[pos_max]

    if X_max:
        max_index = (df[X] >= X_max)
        df[X][max_index] = X_max
    if X_min:
        min_index = (df[X] <= X_min)
        df[X][min_index] = X_min
# 构建中间DataFrame，只保留目标变量"Y"和待转换变量"X"，并进行相应转换

    df_1 = df

    df_1["X_sq"] = df_1[X] ** 2
    df_1["X_sqrt"] = np.sqrt(np.where(df_1[X]<0,0,df_1[X]))  # sqrt(max(x,0))
    df_1["X_cu"] = df_1[X] ** 3
    df_1["X_curt"] = np.cbrt(df_1[X])
    df_1["X_log"] = np.log(np.where(df_1[X] < 1e-3,1e-3,df_1[X]))  # log(max(x,0.001))

    # 从7种转换方法种，挑选最优的转换方法。用"statsmodels"下的Logit，
    # 逐一与y进行建模，并计算每个转化后变量的p_value。
    column=[X, "X_sq","X_sqrt","X_cu","X_curt","X_log",]

    from code_for_model_platform.modeling import stepwise_selection, build_logistic_model
    from sklearn.metrics import roc_auc_score
    model_result = build_logistic_model(Y, column, df_1,seed=2020,maxiter=None)

    min_ix = np.argmin(model_result.pvalues[1:])
    index = model_result.cov_params().index.values[1:]

    feature = index[min_ix]
    auc = roc_auc_score(df_1[Y], model_result.predict())
    if (feature != X) and ( auc > 0.8):
        return  X, feature, df_1[feature]
    return


if __name__ == '__main__':
    df = pd.read_csv('../data/lucheng_data.csv',encoding='gbk').fillna(0)
    t = var_change(df ,'AGE' ,'Y',)
