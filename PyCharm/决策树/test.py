# 导入模块
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# 处理数据集
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.DataFrame(data=iris.target, columns=["label"])


# 特征重命名
map_ = {
    "sepal length (cm)": "花萼长度",
    "sepal width (cm)": "花萼宽度",
    "petal length (cm)": "花瓣长度",
    "petal width (cm)": "花瓣宽度",
}
X = X.rename(columns=map_)


# 对每一列特征进行分箱
X_new = pd.DataFrame()
for c in X.columns:
    binned_data = pd.cut(X[c], bins=4, labels=[0, 1, 2, 3])
    X_new = pd.concat([X_new, binned_data], axis=1)



# 计算信息熵

# 我写的：
def get_entropy_me(y_):
    entropy = -np.sum(
        [
            (i / y_.shape[0]) * np.log2(i / y_.shape[0])
            for i in np.unique(y_, return_counts=True)[1]  # 可用y.value_counts()代替
        ]
    )
    return entropy


# 计算信息增益
# 我写的
def get_information_gain(X_, y_, feature_):
    # 根据给定的feature_取X_
    X_f = X_[feature_]

    # 根据给定的这一列特征不同值, 找到对应的子集index
    indexs = [np.where(X_f == i) for i in np.unique(X_f)]
    for i in np.unique(X_f):
        print(X_f[X_f==i].index)

    # 根据子集index找到子集y, 求子集y的的信息熵并加权求和
    print(X_f)
    print(X_)
    print(y_)
    print(indexs)
    x_entropy = np.sum([get_entropy_me(y_.loc[i]) * (len(y_.loc[i]) / len(y_)) for i in indexs])
    # 返回信息增益
    return get_entropy_me(y_) - x_entropy


# 构建决策树
def build_tree(X_, y_):
    # 如果数据集拆分后，y只有一类了，就停止拆分，返回这一类的标签
    if len(y_.value_counts()) == 1:
        return y_.value_counts().index.tolist()[0][0]

    # 计算最大增益特征
    gains = [get_information_gain(X_, y_, feature) for feature in X_.columns]
    best_feature = X_.columns[np.argmax(gains)]

    # 添加最大信息增益特征到树节点
    tree = {best_feature: {}}

    # 根据最大信息增益特征拆分数据集，获取子集的index
    data_f = X_[best_feature]
    indexs = [np.where(data_f == i)[0] for i in np.unique(data_f)]
    X_ = X_.drop(best_feature, axis=1)
    for i in indexs:
        subset_X = X_.loc[i]
        subset_y = y_.loc[i]
        build_tree(subset_X, subset_y)
    return tree


build_tree(X_new, y)