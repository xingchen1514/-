import numpy as np

def mes_loss(y_true,y_pred):
    """
    这是“损失”LOSS函数
    均方误差
    """
    return((y_true-y_pred)**2).mean()

