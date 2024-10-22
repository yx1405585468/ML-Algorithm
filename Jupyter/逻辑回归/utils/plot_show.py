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

def plot_softmax():
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    x = np.linspace(-2, 2, 100)
    y = softmax(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title('Softmax Function')
    plt.xlabel('Input Value')
    plt.ylabel('Output Probability')
    plt.grid(True)
    plt.show()