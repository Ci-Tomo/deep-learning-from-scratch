# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))  # 親ディレクトリを追加
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

if len(x_train) == 0:
    print("MNIST data is not set up yet.")
    raise SystemExit(0)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)
