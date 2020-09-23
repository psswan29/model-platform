import numpy as np

def build_score_card(model_params, step_score=50):
    return step_score/np.log(2) * model_params

def transfer_score(y_pred, base=300, step_score=50):
    """
    y_pred: y的预测值
    base: 基准分
    step_score: 步长分
    """
    return y_pred.apply(lambda x: np.round(base - np.log(np.where(x/(1 - x) > 0, x/(1 - x), 0.00001)) * step_score / np.log(2)))

if __name__ == '__main__':
    pass