import numpy as np


def cross_entropy_error(y, t):
    print(f"{y=}")
    print(f"{t=}")
    print(f"{y.shape=}")
    print(f"{t.shape=}")
    print(f"{y.size=}")
    print(f"{t.size=}")

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


if __name__ == "__main__":
    y = np.array(
        [
            [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.05],
            [0.1, 0.6, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.05],
        ]
    )
    t = np.array([2, 1])
    print(f"{cross_entropy_error(y, t)=}")
