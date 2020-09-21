# created by SHAOMING, WANG
# on  Sep 10th, 2020
#
from code_for_model_platform import *
from sklearn.model_selection import train_test_split


class Step(object):
    """
    # 此类为之后所有步骤的父类
    """
    def __init__(self, input_nodes=[]):
        # 构建网络的输入节点
        self.input_nodes = input_nodes

        # 重要参数初始化
        # 生成输出节点
        self.output_nodes = []
        # 训练数据集
        self.dataset = None

        # 为了创建网络结构
        for node in self.input_nodes:
            node.output_nodes.append(self)

    # 需要被覆盖
    def forward(self):
        raise NotImplementedError


class Data_input(Step):
    def __init__(self, file_path,encoding,target_n,use_feature=[]):
        """
        # 数据输入:
        # 这里需要一个可视化界面，用户可以选取建模使用的变量
        :param file_path: 文件路径
        :param encoding: 文件编码
        :param target_n: 目标变量名称
        :param use_feature: 使用特征
        """
        super().__init__()
        self.file_path = file_path
        self.encoding = encoding
        self.y_n = target_n
        self.use_feature = use_feature

    def forward(self):
        self.dataset = pd.read_csv(self.file_path,
                                 encoding=self.encoding)
        if not self.use_feature:
            self.X = self.dataset
        else:
            self.X = self.dataset.loc[:,self.use_feature]
        self.Y = self.dataset[self.y_n]


class Preprocess(Step):
    """
    数据预处理， 包括数据清洗、建模测试集划分
    """
    def __init__(self, input_nodes, model='cleanup'):
        super().__init__([input_nodes])
        self.model = model

    def forward(self):
        model_dict = {
            'cleanup':self.__data_cleanup,
            'train_test_split':self.__train_test_split,
            'new_feature': self.__new_feature
        }

        self.X = self.input_nodes[0].X
        self.y = self.input_nodes[0].Y

        self.value = model_dict[self.model]()


    def __data_cleanup(self, x, **kwargs):
        pass

    def __train_test_split(self, **kwargs):
        """
        训练集测试集划分
        :param kwargs:
        :return:
        """
        if 'random_state' in kwargs.keys() and 'test_size' in kwargs.keys() and 'shuffle' in kwargs.keys():
            random_state = kwargs['random_state']
            test_size = kwargs['test_size']
            shuffle = kwargs['shuffle']
        else:
            random_state =123
            test_size = 0.2
            shuffle = False

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, random_state=random_state,
                         test_size=test_size, shuffle=shuffle)
        self.dataset_train, self.dataset_test = train_test_split(self.dataset,random_state=random_state,
                         test_size=test_size, shuffle=shuffle)

    def __new_feature(self,x, **kwargs):
        pass

    def __fill_na(self):
        pass

class Auto_binning(Step):
    def __init__(self):
        super().__init__()
    def forward(self):
        self.X = self.input_nodes[0].X
        self.y = self.input_nodes[0].Y

# 单变量分析
class Univariate_analysis(Step):
    def __init__(self):
        super().__init__()
    def forward(self):
        self.X = self.input_nodes[0].X
        self.y = self.input_nodes[0].Y

# 相关性检验
class Correlation_test(Step):
    """
    相关性检验并降维
    """
    def __init__(self, nodes):
        """

        :param nodes:
        """
        super().__init__([nodes])

    def forward(self):
        # 画图展示相关系数热力图
        self.X = self.input_nodes[0].X
        self.y = self.input_nodes[0].Y

        draw_heat(self.X, self.X.columns)

        # 检验连续变量相关系数
        var_cor_75_dict = tst_continu_var(self.X, self.X.columns)

        # 在这里由于暂时没有合适聚类算法，因此手动筛选变量
        # todo
        list_remove_1 = []
        list_save = []

        for j in var_cor_75_dict.keys():
            list_save.append(j)
            for j2 in var_cor_75_dict[j]:
                if j2[0] >= 0.75 and (j2[1] not in list_save):
                    list_remove_1.append(j2[1])

        self.list_remove = list_remove_1
        self.list_save = list_save



# 建模
class Modeling(Step):
    def __init__(self, nodes, model_method='stepwise'):
        super().__init__([nodes])
        self.model_method = model_method
        self.dataset_train =self.input_nodes[0].dataset_train

    #     todo 设置好属性

    def forward(self):
        if self.model_method == 'stepwise':
            self.model = stepwise_selection(self.dataset_train, sle=0.15, sls=0.15)
        elif self.model_method == 'backwise':
            self.model = backward_selection(self.dataset_train)

        self.predict_y_train = self.model.predict()
        # todo
        self.predict_y_test = self.model.predict(self.X)



# 模型检验
class Model_test(Step):
    def __init__(self, model_nodes):
        super().__init__([model_nodes])
        self.train_y = self.input_nodes[0].train_y
        self.test_y = self.input_nodes[0].test_y
        self.predict_y_train = self.input_nodes[0].predict_y_train
        self.predict_y_test = self.input_nodes[0].predict_y_test
        self.dataset_train = self.input_nodes[0].dataset_train
        self.dataset_test = self.input_nodes[0].dataset_test

    def forward(self):
        # todo 设置好属性
        self.model = self.input_nodes[0].model

        # 模型总结
        self.model_summary = self.model.summary()

        # auc值以及gini系数
        self.AUC, self.GINI = roc_auc_gini(self.train_y, self.predict_y_train)

        # ks值
        self.ks = get_ks(self.predict_y_train, self.train_y)
        self.ks = get_ks(self.predict_y_test, self.test_y)

        # f-1检验
        self.f1_result = f1_test(self.train_y, self.predict_y_train)
        self.f1_result2 = f1_test_m(self.train_y, self.predict_y_train)

        # 模型十等分
        # self.ten_split_train = model_10_split(self.model, self.dataset_train)
        # self.tes_split_test = model_10_split(self.model, self.dataset_test)

        self.ten_split_train = model_10_splitm(self.model, self.dataset_train)
        self.tes_split_test = model_10_splitm(self.model, self.dataset_test)


class Score_card(Step):
    def __init__(self, nodes):
        super().__init__([nodes])
        self.predict_y_train = self.input_nodes[0].predict_y_train
        self.model = self.input_nodes[0].model

    def forward(self):
        print(transfer_score(self.predict_y_train))
        # 模型系数转换为评分卡
        print(build_score_card(self.model.params, step_score=50))

if __name__ == '__main__':
    pass