import numpy as np
from collections import Counter, defaultdict


def validate(x_, y_, ratio=0.2):  # 构造验证集
    n = x_.shape[0]  # 原始数据集的行数
    size = int(n * ratio)  # 验证集的行数

    # 将原始数据集随机排序，生成新的随机序列index
    index_s = np.random.permutation(range(n))

    # 构造多次乱序验证集,生成每一次的验证集的训练集与测试集
    for i in range(int(n / size)):  # 5次

        # 测试集索引
        test_index = index_s[i * size:(i + 1) * size]

        # 除去测试集的索引，剩下的就是训练集的索引
        train_index = list(set(range(n)) - set(test_index))

        # 分批迭代输出训练集与测试集,注意，输出的是一个迭代器
        yield x_[train_index], y_[train_index], x_[test_index], y_[test_index]


class Node:
    def __init__(self, fea=-1, val=None, res=None, right=None, left=None):
        self.fea = fea  # 特征
        self.val = val  # 特征对应的值？分割值？
        self.res = res  # 叶子节点的标签
        self.right = right
        self.left = left


class CartClf:
    def __init__(self, epsilon=1e-3, min_sample=1):
        self.epsilon = epsilon  # 划分的阈值
        self.min_sample = min_sample  # 叶子节点含有的最少样本数
        self.tree = None

    @staticmethod
    def get_gini(y_):
        # 计算整体基尼系数
        c = Counter(y_)  # Counter({0: 50, 1: 50, 2: 50})
        # c.values()  # [50, 50, 50]

        # 基尼系数的计算公式：1-[(50/150)^2+(50/150)^2+(50/150)^2]
        return 1 - sum([(val / y_.shape[0]) ** 2 for val in c.values()])

    def get_fea_gini(self, set1, set2):
        # 计算根据某个特征即相应的特征值切分之后的生成的两个子集的gini系数
        num = set1.shape[0] + set2.shape[0]
        return set1.shape[0] / num * self.get_gini(set1) + set2.shape[0] / num * self.get_gini(set2)

    def best_split(self, split_list, x_, y_):
        """
        遍历所有特征，所有切分点，与切分后的切分子集的gini系数
        选择gini系数最小的切分点，作为最佳切分点，
        如果上一次的gini系数-当前的切分后子集的gini系数<self.epsilon
        则停止切分
        """
        # 返回所有切分点的基尼指数，以字典形式存储。键为split，是一个元组，第一个元素为最优切分特征，第二个为该特征对应的最优切分值
        pre_gini = self.get_gini(y_)  # 当前数据集基尼系数

        # 创建一个包含list的字典，存放切分点以及相应样本点的索引
        sub_data_index = defaultdict(list)  # 创建一个list列表字典存放索引

        # 收集全部的切分特征、切分值和对应的全部索引（统计这样的切分点有多少个重复的，计算gini系数）
        """
        这个地方很难用话解释，举个列子，比如A列有100个值，其中去重之后有3个（5，2，0），那么就有三个切分点，
        每个切分点对应的重复的数据有多少，比如以A列值5为切分点（A列值为5的有50个），统计这50个索引。
        统计50，20，30（假设值为0的有30个），然后分别计算gini系数,
        遍历所有切分点，计算完所有的gini系数后，选择最佳切分点，统计的索引就能生成切分后的子集
        """
        # 这个收集字典的逻辑写的不是很好，比较难懂
        for split in split_list:
            for index, sample in enumerate(x_):  # 遍历所有行数据集，
                if sample[split[0]] == split[1]:  # split[0]是列，split[1]是切分点。
                    sub_data_index[split].append(index)
        min_gini = 1
        best_split = None
        best_set = None
        for split, data_index in sub_data_index.items():
            set1 = y_[data_index]  # 满足切分点条件，为左子树
            set2_index = list(set(range(y_.shape[0])) - set(data_index))
            set2 = y_[set2_index]
            if set1.shape[0] < 1 or set2.shape[0] < 1:
                continue
            now_gini = self.get_fea_gini(set1, set2)
            if now_gini < min_gini:
                min_gini = now_gini
                best_split = split
                best_set = (data_index, set2_index)
        if abs(pre_gini - min_gini) < self.epsilon:  # 若切分后基尼指数下降未超过阈值则停止切分
            best_split = None
        return best_split, best_set, min_gini

    def build_tree(self, split_list, x_, y_):
        """
        循环构造树，子集也会构造树
        """
        # 数据集小于阈值，直接设置为叶子节点
        if y_.shape[0] < self.min_sample:
            return Node(res=Counter(y_).most_common(1)[0][0])
        best_split, best_set, min_gini = self.best_split(split_list, x_, y_)
        if best_split is None:  # 基尼指数下降小于阈值，则终止切分，设为叶节点
            return Node(res=Counter(y_).most_common(1)[0][0])
        else:
            split_list.remove(best_split)
            left = self.build_tree(split_list, x_[best_set[0]], y_[best_set[0]])
            right = self.build_tree(split_list, x_[best_set[1]], y_[best_set[1]])
            return Node(fea=best_split[0], val=best_split[1], right=right, left=left)

    def fit(self, x_, y_):
        """
        Cart分类树与ID3不同，Cart树是一种二叉树，每个节点是特征及对应的某个值组成的元组
        特征可以多次使用
        """
        split_list = []  # [(0, 4.3), (0, 4.4), (0, 4.5)],切分特征及切分点
        for fea in range(x_.shape[1]):  # 对于每个特征
            # 获取每一特征列下独立不重复的值
            unique_vals = np.unique(x_[:, fea])
            if unique_vals.shape[0] < 2:  # 这一列值全部一样
                continue  # 没什么好骚操作的
            elif unique_vals.shape[0] == 2:  # 这一列只有2个特征值,则只有一个切分点，非此即彼,比如5，6，大于5，或者小于5就能切分了
                split_list.append((fea, unique_vals[0]))
            else:
                for val in unique_vals:
                    split_list.append((fea, val))

        self.tree = self.build_tree(split_list, x_, y_)

    def predict(self, x_):
        def helper(x_, tree):
            if tree.res is not None:  # 表明到达叶节点
                return tree.res
            else:
                if x_[tree.fea] == tree.val:  # "是" 返回左子树
                    branch = tree.left
                else:
                    branch = tree.right
                return helper(x_, branch)

        return helper(x_, self.tree)

    def disp_tree(self):
        # 打印树
        self.disp_helper(self.tree)

    def disp_helper(self, current_node):
        # 前序遍历
        print(current_node.fea, current_node.val, current_node.res)
        if current_node.res is not None:
            return
        self.disp_helper(current_node.left)
        self.disp_helper(current_node.right)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x = load_iris().data
    y = load_iris().target

    # 生成一个验证集的迭代器
    iter_data = validate(x, y, ratio=0.2)

    # 交叉验证
    for item in iter_data:
        score = 0  # 计数器
        x_train, y_train, x_test, y_test = item
        clf = CartClf()  # cart分类树
        clf.fit(x_train, y_train)
        for x, y, in zip(x_test, y_test):
            if clf.predict(x) == y:  # 预测正确
                score += 1  # 分数+1

        # 正确预测数/总数，得出预测准确率
        print(score / len(y_test))
