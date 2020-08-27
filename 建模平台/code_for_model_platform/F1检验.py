from sklearn.metrics import classification_report
import numpy as np

def f1_test(y_train, y_pred):
    """

    :param y_train:
    :param y_pred:
    :return:
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