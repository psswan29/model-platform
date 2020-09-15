# 等分表及洛伦兹曲线
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def cross_table(var, y, tie=20):
    '''
    #cross_table函数功能描述：根据选择的分析变量，目标变量及等分的层数，生成各层的统计及累计指标计算
    #输入
    #    1.var:要进行分层的变量，例如：data[xxx]，需要为数值型变量，最好没有缺失
    #    2.y:目标变量，例如：data[xxx]。要求是0/1编码的数值格式
    #    3.tie：需要分的层数，默认20层。要为大于0的整数
    #输出
    #    1.按照层数的统计表：有按照层的统计指标，观测数，最大/最小，均值，y=1数量，y=1占比
    #    2.绘制洛伦兹曲线

    #使用该函数需要 使用如下三个对象 pd,np,plt
    '''
    table = pd.DataFrame({"var": var, "y": y, })

    # n为总记录数
    n = len(table)

    # 计算分层变量，起始点为1
    lst = []
    for i in range(n):
        lst = lst + [int(i * tie / n) + 1]

    # 对分析变量进行升序（由小到大）排序
    temp = table.sort_values("var", axis=0, ascending=True)

    # 将分层变量并入排序好的表temp中，形成后面统计的基表
    temp1 = pd.DataFrame({"rank": lst, "var": temp["var"], "y": temp["y"], })

    # 在基表的基础上，进行汇总统计生成grouped表
    grouped = pd.DataFrame(index=np.unique(temp1['rank']))
    grouped["freq"] = temp1.groupby(temp1["rank"])["var"].count()  # 每层观测数
    grouped["min"] = temp1.groupby(temp1["rank"])["var"].apply(lambda a: min(a))  # 每层最小观测值
    grouped["max"] = temp1.groupby(temp1["rank"])["var"].apply(lambda a: max(a))  # 每层最大观测值
    grouped["mean"] = temp1.groupby(temp1["rank"])["var"].mean()  # 每层测值均值
    grouped["y=1个数"] = temp1.groupby(temp1["rank"])["y"].sum()  # 每层y=1数量
    grouped["y=1占比(%)"] = 100 * temp1.groupby(temp1["rank"])["y"].mean()  # 每层y=1占比

    # 计算y=1累计占比
    grouped["y=1的总占比(%)"] = 100 * temp1.groupby(temp1["rank"])["y"].sum() / temp1["y"].sum()  # 每层y=1总占比
    n_group = len(grouped["y=1的总占比(%)"])
    array_y = np.array(grouped["y=1的总占比(%)"])
    accu = []
    sum1 = 0
    for i in range(n_group):
        if i == 0:
            sum1 = array_y[i]
            accu = accu + [sum1]
        else:
            sum1 = sum1 + array_y[i]
            accu = accu + [sum1]
    grouped["Y=1累积占比(%)"] = accu

    # 计算每层记录数量累计占比
    array_y = np.array(grouped["freq"])
    accu1 = []
    sum1 = 0
    for i in range(n_group):
        if i == 0:
            sum1 = 100 * array_y[i] / n
            accu1 = accu1 + [sum1]
        else:
            sum1 = sum1 + 100 * array_y[i] / n
            accu1 = accu1 + [sum1]
    grouped["数量累积占比(%)"] = accu1
    plot_roc_curve(grouped)

    # print(grouped)
    return grouped

def plot_roc_curve(grouped):
    # 绘制洛伦兹曲线
    plt.plot(grouped["数量累积占比(%)"], grouped["数量累积占比(%)"], label="未使用模型")
    plt.plot(grouped["数量累积占比(%)"], grouped["Y=1累积占比(%)"], label="使用模型")
    plt.grid(True)
    plt.legend()
    plt.show()
    print("")