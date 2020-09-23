import seaborn as sns
import matplotlib.pyplot as plt

def visualize_continua_var_layers(var_type_dict, data_1):
    """
    可视化连续变量层级, 通过箱型图展现
    :param var_type_dict:
    :param data_1:
    :return:
    """
    for i in var_type_dict['continua']:
        fig, ax = plt.subplots()  # 创建子图
        sns.boxplot(data=data_1[i], palette=['m'])
        ax.set(title=i)
        plt.show()
    return