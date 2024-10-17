import numpy as np
import matplotlib.pyplot as plt

def plot_sigmoid():
    # 定义Sigmoid函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # 生成-10到10之间的一系列数字作为x值
    x = np.linspace(-10, 10, 100)

    # 计算对应的Sigmoid函数值
    y = sigmoid(x)

    # 绘制Sigmoid函数图表
    plt.figure()
    plt.plot(x, y, label='Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.title('Sigmoid Function')
    plt.grid(True)
    plt.legend()
    plt.show()
