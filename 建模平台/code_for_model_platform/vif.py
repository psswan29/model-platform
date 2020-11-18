import pandas as pd
import numpy as np

def calculate_vif(X, threshold=100):
    """
    计算vif值进行多重共线性检验
    :param X:
    :param threshold:
    :return:
    1. 一个dataframe, columns 包括 变量名，vif， vif检测是否大于设定阈值threshold
    2. vif最大的一个变量名
    3. 未通过vif值检验的变量

    在不存在多重共线性的情况下，方差扩大因子接近于1。
    但是，实际上自变量之间总是或多或少地存在多重共线性，
    因而将方差扩大因子等于1作为评价共线性的标准是不现实的。
    多重共线性越强，方差扩大因子就越大。一个易用的标准：
    当VIF值大于10时，就认为变量之间具有强烈的多重共线性，不能接受。"""

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

    yy=pd.DataFrame({'var_name': X.columns, 'vif':vif})
    yy['vif_test'] = yy.vif >= threshold
    pos = np.argmax(yy['vif'])

    return yy, yy['var_name'][pos], yy['var_name'][yy['vif_test']]


if __name__ == '__main__':
    df = pd.DataFrame([[15.9, 16.4, 19, 19.1, 18.8, 20.4, 22.7, 26.5, 28.1, 27.6, 26.3]
                          , [149.3, 161.2, 171.5, 175.5, 180.8, 190.7, 202.1, 212.1, 226.1, 231.9, 239]
                          , [4.2, 4.1, 3.1, 3.1, 1.1, 2.2, 2.1, 5.6, 5, 5.1, 0.7]
                          , [108.1, 114.8, 123.2, 126.9, 132.1, 137.7, 146, 154.1, 162.3, 164.3, 167.6]]).T
    columns = ["var1", "var2", "var3", "var4"]
    df.columns = columns
    yy = calulate_vif(df)