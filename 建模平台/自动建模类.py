# created by SHAOMING, WNAG
# on  Sep 10th, 2020

from code_for_model_platform import *
from sklearn.model_selection import train_test_split

# 此类为之后所有步骤的父类
class Step(object):
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.X = None

        # 为了创建网络结构
        for node in self.input_nodes:
            node.output_nodes.append(self)

    # 需要被覆盖
    def forward(self):
        raise NotImplementedError
    # 需要被覆盖
    def backward(self):
        raise NotImplementedError

# 数据输入:
# 这里需要一个可视化界面，用户可以选取建模使用的变量
class Data_input(Step):
    def __init__(self, file_path,encoding,use_feature=[]):
        super(Step, self).__init__()
        self.file_path = file_path
        self.encoding = encoding
        self.use_feature = use_feature

    def forward(self):
        self.X = pd.read_csv(self.file_path,
                                 encoding=self.encoding,
                                 usecols=self.use_feature)
    def backward(self):
        pass

# 数据预处理， 包括数据清洗、建模测试集划分
class Preprocess(Step):
    def __init__(self, X, skip=False):
        super(Step, self).__init__([X])
        self.skip = skip
    def forward(self):
        pass
    def backward(self):
        pass
    def _data_cleanup(self):
        pass
    def _train_test_split(self):
        pass
    def _new_feature(self):
        pass

# 单变量分析
class Univariate_analysis(Step):
    def __init__(self):
        super(Step, self).__init__()
    def forward(self):
        pass
    def backward(self):
        pass

# 相关性检验
class Correlation_test(Step):
    def __init__(self):
        super(Step, self).__init__()
    def forward(self):
        pass
    def backward(self):
        pass


# 建模
class Modeling(Step):
    def __init__(self):
        super(Step, self).__init__()
    def forward(self):
        pass
    def backward(self):
        pass


# 模型检验
class Model_test(Step):
    def __init__(self):
        super(Step, self).__init__()

    def forward(self):
        pass

    def backward(self):
        pass


