from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
def f1_test(y_train, y_pred):
    """
    version 邓欣
    :param y_train: 训练集目标变量，真值
    :param y_pred: 预测目标变量，预测值
    :return: 一个列表
    """
    f1 = []
    i = 0
    while i < 1:
        y_pred1 = []
        for j in range(len(y_pred)):
            if y_pred.values[j, 0] > i:
                y_pred1.append(1)
            elif y_pred.values[j, 0] <= i:
                y_pred1.append(0)
        y_pred2 = np.array(y_pred1)
        i += 0.05
        f1_score = classification_report(y_train, y_pred2, target_names=['up', 'down'])
        f1.append([i, f1_score])

    print(np.array(f1))
    return f1

def f1_test_m(y_train, y_pred):
    """
    created by SHAOMING WANG
    :param y_train: 训练集目标变量，真值
    :param y_pred:  预测目标变量，预测值
    :return:
    """
    from collections import Counter
    c = Counter(y_train)
    assert len(c.keys()) == 2

    temp_df = pd.DataFrame(y_train)
    temp_df.columns = ['y_train']
    threshold = np.sum((y_train == 1).astype(int))/y_train.shape[0]
    temp_df['y_pred'] = (y_pred >= threshold).astype(int)
    f1_score = classification_report(temp_df['y_train'], temp_df['y_pred'])
    for i in f1_score.split('\n'):
        print(i)
    return f1_score
