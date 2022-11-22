import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize(data_):  # 归一标准化
    # 转化为浮点数
    data_ = data_.astype(float)

    # 根据浮点数计算均值
    data_mean = np.mean(data_, axis=0)

    # 计算标准差
    data_deviation = np.std(data_, axis=0)

    # 进行标准化操作
    if data_.shape[0] > 1:
        data_ = data_ - data_mean

    # 防止除以0
    data_deviation[data_deviation == 0] = 1
    data_ = data_ / data_deviation

    return data_


if __name__ == '__main__':
    # TODO 1: 获取数据集 x , y
    data = pd.read_csv('../../../线性回归/data/non-linear-regression-x-y.csv')
    x = data['x'].values.reshape((data.shape[0], 1))
    y = data['y'].values.reshape((data.shape[0], 1))
    # plt.plot(x, y)
    # plt.show()

    # TODO 2: 标准化预处理
    x = normalize(x)
    print(x)
