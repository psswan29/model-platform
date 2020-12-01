import pandas as pd
import numpy as np

def cal_gini(y1, y2):
    """
    计算基尼系数
    y1: 因变量子集1
    y2：因变量子集2
    """

    len_y1 = len(y1)
    len_y2 = len(y2)
    sum_y =  len(y1) + len(y2)

    p1 = np.mean(y1) if not len(y1) else 0
    p2 = np.mean(y2) if not len(y2) else 0
    return (len_y1/sum_y) * (2*p1*(1-p1)) + (len_y2/sum_y)*(2*p2*(1-p2))


def ginni_split_con_var(x, y):
    """
    使用ginni系数划分连续变量
    x：需要进行分层的变量
    y：目标变量，必须是0,1变量
    """
    # 连续变量基尼系数划分，cart树划分，对连续变量的分箱
    # 只能进行二分法划分

    df = pd.DataFrame(zip(x, y), columns=['x', 'y'])
    df['x'].fillna(df.x.mean(),inplace=True)
    #     遍历x中所有值
    #    求每种值对应的gini系数
    #    排序，输出最小的两款
    gini_list = sorted([(i, cal_gini(df['y'][df.x < i + 0.001], df['y'][df.x >= i + 0.001])) for i in set(df['x'].values)],
                       key=lambda x: x[1])
    return gini_list[:2]

def ginni_split_cat_var(x, y):
    """
    使用ginni系数划分类别变量
    x：需要进行分层的变量
    y：目标变量，必须是0,1变量
    """

    #     类别变量基尼系数划分，cart树划分，对类别变量的分箱
    # 只能进行二分法划分，类似于对连续变量的划分
    df = pd.DataFrame(zip(x, y), columns=['x', 'y'])

    gini_list = sorted([(i, cal_gini(df['y'][df.x == i], df['y'][df.x != i])) for i in set(x)], key=lambda x: x[1])
    return gini_list[:2]


def gini_split(df, cate_var=[], conti_var=[], y_name='y', gap=0.02):
    # 初始化字典，用于保存分箱信息
    split_dict = {}
    #     遍历df除去y外，所有变量
    for col in [col for col in df.columns if col not in [y_name]]:
        #         判断变量类型
        #         若为类别变量
        if col in cate_var:
            split_info = ginni_split_cat_var(df[col], df[y_name])
        #         若为连续变量
        elif col in conti_var:
            split_info = ginni_split_con_var(df[col], df[y_name])
        else:
            continue
        #         返回一个列表，列表中是两个二项元组，
        #         每个元组第一项为分箱值，第二项为gini系数（约大代表混乱度越高）
        #         当这两个元组的gini值相差小于gap值时，同时保留splitinfo中第一项与第二项
        #         否则只保留第一项
        if (split_info[1][1] - split_info[0][1]) <= gap:
            split_dict[col] = [e[0] for e in split_info]
        else:
            split_dict[col] = [split_info[0][0]]

    return split_dict


if __name__ == '__main__':
    print(cal_gini([0,0,0,0,1,0,1],[1,1,0]))
    print(ginni_split_con_var(np.arange(10), [0, 0, 0, 0, 1, 0, 1, 1, 1, 0]))
    print(ginni_split_cat_var([0,1,1,0,0,2,0,0,1,1],[0,0,0,0,1,0,1,1,1,0]))
    df = pd.read_csv('../lucheng_data.csv',encoding='gbk')
    print(gini_split(df,conti_var=['AGE'],y_name='Y',gap=0.05))



