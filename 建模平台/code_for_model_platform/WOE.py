import pandas as pd

def WOE_IV_encoding(df, col, target):
    """
    计算某个变量的WOE值，IV值
    :param df: 源数据, pandas 的dataframe格式
    :param col:  用来计算iv值和WOE值的变量名
    :param target: 目标变量
    :return: 一个字典
    """
    total = df.groupby([col]).agg({target:['count', 'sum']})
    total.columns = ['total', 'target']

    total = pd.DataFrame({'total':total.count()})
    targeted_ = total.sum()
    targeted_ = pd.DataFrame({'targeted': targeted_})

    total.merge(targeted_,how='left',left_index=True, right_index=True)









