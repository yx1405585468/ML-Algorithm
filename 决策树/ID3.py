from math import log
from sklearn.datasets import load_iris
import pandas as pd

pd.set_option("display.width", None)


# 定义二叉树节点
class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature_index=None):
        """
        :param root: 当前节点是否为根节点，（也就是最下层的是叶子节点，也是最终的决策结果）
        :param label: 当前节点的标签
        :param feature_name: 当前节点的特征名称
        :param feature_index: 当前节点的特征
        """
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature_index
        self.tree = {}  # 字典结构，可添加节点
        self.result = {  # 决策树的结构
            "label": self.label,
            "feature": self.feature,
            "tree": self.tree
        }

    def __repr__(self):
        return "{}".format(self.result)

    def add_node(self, value, node):
        self.tree[value] = node  # 添加节点

    def predict(self, features):
        if self.root is True:  # 如果已经是根节点（最下层的叶子节点），直接返回预测结果，也就是label
            return self.label  # 叶子节点是带有标签的
        return self.tree[features[self.feature]].predict(features)  # 不是叶子节点，就划分到下一个节点来判断


class DTree:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self._tree = {}

    # dataframe计算熵的方式，如若输入的是整个data，则计算的就是数据集的整体经验熵
    @staticmethod
    def calculate_entropy(data_):
        data_length = data_.shape[0]  # 数据集data的行数，iris数据集为150行
        target = data_.iloc[:, -1]  # 数据集data的标签列
        classes_count = list(target.value_counts())  # data中每一类标签的数量。如iris数据集标签[0,1,2]的数量分别为[50,50,50]
        entropy = -sum([(p / data_length) * log(p / data_length, 2) for p in classes_count])  # 计算整体数据集的熵
        return entropy

    # 计算特征对数据集data的经验条件熵
    def conditional_entropy(self, data_, i):  # i是特征列
        data_length = data_.shape[0]  # 数据集data的行数，iris数据集为150行
        feature_sets = {}
        for j in range(data_length):
            feature = data_.iloc[j, i]  # 数据集data的第i列的所有行
            if feature not in feature_sets:
                feature_sets[feature] = []  # 为第i列的所有行数据创建一个列表

            # 统计第i列的每一个数有多少行数据，第i列数有重复的，把重复数据的多行添加在一起
            feature_sets[feature].append(data_.iloc[j, :])
            # 输入第i列的每一个数据组成的dataframe,分别计算这些dataframe的熵，求和得到这一整列的经验条件熵
        cond_ent = sum(
            [(len(p) / data_length) * self.calculate_entropy(pd.DataFrame(p)) for p in feature_sets.values()])
        return cond_ent

    # 计算信息增益
    def information_gain(self, data_):
        count = data_.shape[1] - 1  # 计算数据集data的列数，即特征数，iris中为4个
        entropy = self.calculate_entropy(data_)  # 输入的是整个数据集，所以计算得到整个数据集data的整体经验熵
        best_feature = []
        # 通过循环遍历的方式，遍历4列（4个特征），来计算每一个特征的经验条件熵
        for i in range(count):  # 遍历4列，计算每一列的信息增益
            c_info_gain = entropy - self.conditional_entropy(data_, i)  # 计算第i列的信息增益（c=1,2,3,4）
            best_feature.append((i, c_info_gain))  # 将当前特征的位置与信息增益联系在一起
        # 比较大小
        # max函数会比较list中的每一组元素（当前四组），而key会选择四组中的哪个元素进行比较，lambda x: x[1]即选择第二个元素进行比较
        best_ = max(best_feature, key=lambda x_: x_[1])
        return best_

    def train(self, data_):
        # 获取当前训练集的数据_,标签y_train,训练集的特征features(不好含target)
        _, y_train, features = data_.iloc[:, :-1], data_.iloc[:, -1], data_.columns[:-1]

        # 若当前训练集中只有一类标签了，意味着决策树当前分支生长结束，
        if len(y_train.value_counts()) == 1:
            # 生成最终的叶子节点（在这里就是根节点）,标签就为当前数据集的标签
            node = Node(root=True, label=y_train.iloc[0])
            return node

        # 若特征也分配完，则没搞懂
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 计算最大信息增益，并选出信息增益最大的特征
        feature_index, max_info_gain = self.information_gain(data_)
        max_feature_name = features[feature_index]  # 通过特征的index找到其特征名

        # 若当前特征的信息增益小于阈值threshold,则
        if max_info_gain < self.threshold:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 用最大信息增益特征构建决策树的节点,生成决策树
        node_tree = Node(root=False, feature_name=max_feature_name, feature_index=feature_index)

        # 当前特征列的不重复值
        feature_list = data_[max_feature_name].value_counts().index

        # 由于当前列为当前最大信息增益特征,已被用来构建决策树的分支节点,所以要将此特征列从现有数据集中剔除
        for f in feature_list:
            sub_train_df = data_.loc[data_[max_feature_name] == f].drop([max_feature_name], axis=1)

            # 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        return node_tree
        # 昨天截至到这里,今天从这里开始
        exit(0)

    def fit(self, x_train, y_train):
        train_data = pd.concat([x_train, y_train], axis=1)
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, x_test):
        return x_test.apply(lambda x_: self._tree.predict(x_), axis=1)


if __name__ == '__main__':
    # 下载iris数据集，并转换为dataframe数据结构
    iris = load_iris()
    x = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    y = pd.DataFrame(data=iris.target, columns=["target"])

    # 生成决策树对象
    dt = DTree()

    # fit训练集
    tree = dt.fit(x, y)

    # 预测数据
    result = dt.predict(x)
    print(result)
