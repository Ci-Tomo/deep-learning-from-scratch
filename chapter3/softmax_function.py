import numpy as np


def softmax(x):
    x = np.asarray(x)
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)


if __name__ == "__main__":
    x = np.array([0.3, 2.9, 4.0])
    y = softmax(x)
    print(y)
    print(np.sum(y))
