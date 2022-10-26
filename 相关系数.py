import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

pd.set_option("display.width", None)
if __name__ == '__main__':
    # 1. corr相关系数
    iris = load_iris()
    x = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    cor_matrix = x.corr()
    # print(cormatrix)

    # 2. 转化为上三角矩阵显示
    cor_matrix *= np.tri(*cor_matrix.values.shape, k=-1).T
    # print(cor_matrix)

    # 3. 整合处理,每个特征与其他特征之间的相关性
    cor_matrix = cor_matrix.stack()
    # print(cor_matrix)

    # 4. 排序
    cor_matrix = cor_matrix.reindex(cor_matrix.abs().sort_values(ascending=False).index).reset_index()
    print(cor_matrix)
