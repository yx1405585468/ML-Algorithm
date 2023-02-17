import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier


def plot_digit(data):
    image = data.reshape(8, 8)
    plt.imshow(image, cmap=matplotlib.cm.hot)
    plt.axis('off')


if __name__ == '__main__':
    mnist = load_digits()
    rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rfc.fit(mnist['data'], mnist['target'])
    print(rfc.feature_importances_.shape)

    plot_digit(rfc.feature_importances_)
    char = plt.colorbar(ticks=[rfc.feature_importances_.min(), rfc.feature_importances_.max()])
    char.ax.set_yticklabels(["Not important", "Very important"])
    plt.show()
