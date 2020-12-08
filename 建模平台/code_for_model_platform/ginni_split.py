import pandas as pd
import numpy as np
import icecream as ic

def cal_gini(y1, y2):
    """
    计算基尼系数
    y1: 因变量子集1
    y2：因变量子集2
    """
    # print(y1)
    # print(y2)
    len_y1 = len(y1)
    len_y2 = len(y2)
    # ic.ic(len_y1)
    # ic.ic(len_y2)

    sum_y =  len(y1) + len(y2)

    if not len_y1:
        p1 =0
    else:
        p1 = np.mean(y1)
    # print(p1)
    if not len_y2:
        p2 =0
    else:
        p2 = np.mean(y2)
    # print(p2)

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
    return gini_list

def ginni_split_cat_var(x, y):
    """
    使用ginni系数划分类别变量
    x：需要进行分层的变量
    y：目标变量，必须是0,1变量
    """

    #     类别变量基尼系数划分，cart树划分，对类别变量的分箱
    # 只能进行二分法划分，类似于对连续变量的划分
    df = pd.DataFrame(zip(x, y), columns=['x', 'y'])

    unique_values = set(x)
    gini_list = sorted([(i, cal_gini(df['y'][df.x == i], df['y'][df.x != i])) for i in unique_values],
                       key=lambda x: x[1])

    if len(unique_values)==2:
        return gini_list[:1]
    return gini_list


def gini_split(df, cate_var=[], conti_var=[], y_name='y', rate=0.1,gaps=0.01):
    # 初始化字典，用于保存分箱信息
    split_dict = {}
    #     遍历df除去y外，所有变量
    for col in [col for col in df.columns if col not in [y_name]]:
        #         判断变量类型
        #         若为类别变量
        __cate_flag = False
        if col in cate_var:
            split_info = ginni_split_cat_var(df[col], df[y_name])
            __cate_flag = True
        #         若为连续变量
        elif col in conti_var:
            split_info = ginni_split_con_var(df[col], df[y_name])
        else:
            continue
        #         返回一个列表，列表中是两个二项元组，
        #         每个元组第一项为分箱值，第二项为gini系数（约大代表混乱度越高）
        # print(split_info)

        split_values = [v for k,v in split_info]
        split_keys = [k for k,v in split_info]
        # print(split_info)
        threshold = df.shape[0] * rate
        # print(threshold)
        if __cate_flag:
            split_info_gaps = [(split_values[i]-split_values[i+1]) for i in range(len(split_values)-1)]

            for i in range(len(split_keys[:-1])):
                num = np.sum(df[col].isin(split_keys[:i + 1]))
                if threshold>= num:
                    continue
                else:
                    split_dict[col] = split_keys[:i+1]
                    break
        else:
            split_dict[col] = split_keys[0]
    return split_dict


if __name__ == '__main__':
    # print(cal_gini([0,0,0,0,1,0,1],[1,1,0]))

    # print(ginni_split_con_var(np.arange(10), [0, 0, 0, 0, 1, 0, 1, 1, 1, 0]))
    # print(ginni_split_cat_var([0,1,1,0,0,2,0,0,1,1],[0,0,0,0,1,0,1,1,1,0]))
    df = pd.read_csv('../lucheng_data.csv',encoding='gbk')
    df.select_dtypes(object).fillna('nan',inplace=True)
    df.fillna(0,inplace=True)
    # print(gini_split(df,conti_var=['AGE'],y_name='Y',gaps=0.05))
    print(gini_split(df, cate_var=['PAY_MODE'], y_name='Y', gaps=0.05))
    cate_vars = ['GENDER', 'PAY_MODE', 'SERVICE_TYPE',
                 'GROUP_FLAG', 'USER_STATUS', 'FACTORY_DESC',
                 'REAL_HOME_FLAG_M1',
                 'LIKE_HOME_FLAG_M1', 'REAL_WORK_FLAG_M1',
                 'LIKE_WORK_FLAG_M1']

    print(gini_split(df, cate_var=cate_vars, y_name='Y', gaps=0.05))
    gini_split(df, cate_var=['GROUP_FLAG'], y_name='Y')


