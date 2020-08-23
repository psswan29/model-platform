#划分平衡数据集
import pandas as pd
from sklearn.model_selection import train_test_split


def data_balance_split(data,Y,y_0,y_1,seed=12345,Y_0=0,Y_1=1):
    '''
    data:数据集
    Y:目标变量名称
    y_0:非目标事件抽样比例
    y_1:目标事件抽样比例
    seed：随机数种子
    Y_0:非目标事件标签
    Y_1:目标事件标签
    '''
    data_X = data[data[Y]==Y_0]  #非目标事件集合
    data_Y = data[data[Y]==Y_1]  #目标事件集合
    X_train,X_test = train_test_split(data_X, test_size=y_0, random_state=seed)
    if y_1 < 1:
        Y_train,Y_test = train_test_split(data_Y, test_size=y_1, random_state=seed)
    elif y_1 == 1:
        Y_test = data_Y
    Test = pd.concat([X_test,Y_test],axis=0)
    return Test