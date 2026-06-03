import os
import urllib.request

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_DIR = os.path.dirname(__file__)
MNIST_PATH = os.path.join(DATA_DIR, "mnist.npz")


def _download_mnist(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(MNIST_URL, path)


def _to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(MNIST_PATH):
        _download_mnist(MNIST_PATH)

    with np.load(MNIST_PATH) as f:
        x_train = f["x_train"]
        t_train = f["y_train"]
        x_test = f["x_test"]
        t_test = f["y_test"]

    if flatten:
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if one_hot_label:
        t_train = _to_one_hot(t_train)
        t_test = _to_one_hot(t_test)

    return (x_train, t_train), (x_test, t_test)
