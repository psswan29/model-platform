#logit/elogit图
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def drawing(data_tt,X1,Y1, N=20, max_f=None,min_f=None, DF=False, skip=None):
    '''
    data_tt:数据集
    X1:因变量
    Y1:目标变量
    N:要分的层数
    DF:False或者True,False代表：不等分；True代表：等分
    max_f:最大值
    min_f：最小值
    '''
    x = data_tt[X1]
    y = data_tt[Y1]
    x[x >= max_f] = max_f
    x[x <= min_f] = min_f

    n = len(y)  # 计算观测的个数
    group = N  # 组数
    temp = pd.DataFrame({'x': x, 'y': y, })  # 生成临时表

    # 给x排序
    temp1 = temp.sort_values('x', axis=0, ascending=True)

    # 构建频次统计标记[2,:]

    # 计算mean分组方法的顺序号。
    # 先计算同一个“x”最小的顺序号
    temp1['s'] = 0
    for j in range(n):
        if j == 0:
          temp1.iat[j, 2] = j + 1
        else:
            if temp1.iat[j, 0] == temp1.iat[j - 1, 0]:
              temp1.iat[j, 2] = temp1.iat[j - 1, 2]
            else:
              temp1.iat[j, 2] = j + 1
            pass
        pass
    pass
    temp2 = temp1.sort_values('x', axis=0, ascending=False)
    # 再计算同一个“x”最大的顺序号
    temp2['m'] = 0
    for k in range(n):
        if k == 0:
          temp2.iat[k, 3] = n - k
        else:

            if temp2.iat[k, 0] == temp2.iat[k - 1, 0]:
              temp2.iat[k, 3] = temp2.iat[k - 1, 3]
            else:
              temp2.iat[k, 3] = n - k
            pass
        pass
    pass
    # 而后求均值
    temp3 = temp2.sort_values('x', axis=0, ascending=True)
    temp3['t'] = (temp3['s'] + temp3['m']) / 2



    if DF == False:
        # 通过mean分组方法的顺序号进行分组
        temp3['RK'] = 0
        for l in range(n):
          temp3.iat[l, 5] = (temp3.iat[l, 4] * N) / (n + 1)
       #   print(temp3.iat[l, 5])
        grouped = pd.DataFrame(index=np.unique(temp3['RK']))
        #print(np.unique(temp3['RK']))
        # 每层最小观测值
        grouped["min"] = temp3.groupby(temp3["RK"])['x'].apply(lambda a: min(a))
        # 每层最大观测值
        grouped["max"] = temp3.groupby(temp3["RK"])['x'].apply(lambda a: max(a))
        # 每层测值均值
        grouped["mean"] = temp3.groupby(temp3["RK"])['x'].mean()
        # 每层y=1占比
        grouped["p"] = temp3.groupby(temp3["RK"])['y'].mean()
        # 每层观测数
        grouped["freq"] = temp3.groupby(temp3["RK"])['x'].count()
        # logit
        grouped["logit"] = np.log(grouped["p"] / (1 - grouped["p"]))
        # elogit
        grouped["elogit"] = np.log(
            (grouped["p"] + 1 / np.sqrt(2 * grouped["freq"]))
            / (1 - grouped["p"] + 1 / np.sqrt(2 * grouped["freq"]))
        )
    elif DF == True:
        temp3['rank_1'] = range(len(temp3))
        # 将data_sorted.rank_1分成20fen(0-19)
        temp3['RK'] = pd.qcut(temp3.rank_1, group, labels=False)
        grouped = pd.DataFrame(index=np.unique(temp3['RK']))
        # 每层最小观测值
        grouped["min"] = temp3.groupby(temp3['RK'])['x'].apply(lambda a: min(a))
        # 每层最大观测值
        grouped["max"] = temp3.groupby(temp3['RK'])['x'].apply(lambda a: max(a))
        # 每层测值均值
        grouped["mean"] = temp3.groupby(temp3['RK'])['x'].mean()
        # 每层y=1占比
        grouped["p"] = temp3.groupby(temp3['RK'])['y'].mean()
        # 每层观测数
        grouped["freq"] = temp3.groupby(temp3['RK'])['x'].count()
        # logit
        grouped["logit"] = np.log(grouped["p"] / (1 - grouped["p"]))
        # elogit
        grouped["elogit"] = np.log((grouped["p"] + 1 / np.sqrt(2 * grouped["freq"])) / (
        1 - grouped["p"] + 1 / np.sqrt(2 * grouped["freq"])))
    else:
        print('请确定您所输入的DF值是否为True或False！')

    # 作图
    print(grouped)
    '''
    plt.xlabel(X1)
    plt.ylabel(Y1)   
    plt.plot(grouped["mean"], grouped["logit"], label="logit")  # logit曲线
    plt.plot(grouped["mean"], grouped["elogit"], label="elogit")  # elogit曲线
    plt.legend()
    plt.show()
    '''
    if not skip:
        sns.lmplot(x="mean", y="logit", data=grouped[:],fit_reg=False)
        plt.title(X1 + '_LOGIT')
        plt.show()
        sns.lmplot(x="mean", y="elogit", data=grouped,fit_reg=False)
        plt.title(X1 + '_ELOGIT')
        plt.show()
    else:
        sns.lmplot(x="mean", y="logit", data=grouped[:-skip],fit_reg=False)
        plt.title(X1 + '_LOGIT')
        plt.show()
        sns.lmplot(x="mean", y="elogit", data=grouped[:-skip],fit_reg=False)
        plt.title(X1 + '_ELOGIT')
        plt.show()
    print("")
    return grouped