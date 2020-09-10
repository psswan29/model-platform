import numpy as np

class Step(object):
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.value = None

        for node in self.output_nodes:
            node.input_nodes.append(self)
    # 需要被覆盖
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class data_input(Step):
    def __init__(self):
        super(Step,self).__init__()

