#划分训练、测试数据集
import pandas as pd
from sklearn.model_selection import train_test_split


def data_train_split(data,Y,r_0,seed=12345,Y_0=0,Y_1=1):
    '''
    data:数据集
    Y:目标变量名称
    r_0:测试集占比
    seed：随机数种子
    Y_0:非目标事件标签
    Y_1:目标事件标签
    '''
    data_X = data[data[Y]==Y_0]  #非目标事件集合
    data_Y = data[data[Y]==Y_1]  #目标事件集合
    X_train,X_test = train_test_split(data_X, test_size=r_0, random_state=seed)
    Y_train,Y_test = train_test_split(data_Y, test_size=r_0, random_state=seed)
    Train = pd.concat([X_train,Y_train])
    Test = pd.concat([X_test,Y_test])
    return Train,Test