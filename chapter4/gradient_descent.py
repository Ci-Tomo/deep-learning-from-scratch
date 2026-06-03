import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from numerical_differentiation import function_2, numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


if __name__ == "__main__":
    init_x = np.array([-3.0, 4.0])
    print(f"{gradient_descent(function_2, init_x, lr=0.1, step_num=100)=}")
