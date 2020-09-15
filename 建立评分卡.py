import numpy as np

def build_score_card(model_params, step_score=50):
    """
    建立评分卡
    :param model_params: 模型参数
    :param step_score: 步长分
    :return:
    一个列表
    """
    return step_score/np.log(2) * model_params

def transfer_score(y_pred, base=300, step_score=50):
    """
    评分卡评分转换
    :param y_pred: y的预测值
    :param base: 基准分
    :param step_score: 步长分
    :return:
    一个numpy 数组
    """
    return y_pred.apply(lambda x: np.round(base - np.log(np.where(x/(1 - x) > 0, x/(1 - x), 0.00001)) * step_score / np.log(2)))

if __name__ == '__main__':
    pass