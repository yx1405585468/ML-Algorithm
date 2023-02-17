if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x = load_iris().data
    y = load_iris().target

    for index, sample in enumerate(x):
        print(sample)
        exit()
