import os
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 选择哪一块gpu,如果是-1，就是调用cpu
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import tensorflow as tf
import tensorflow.keras as keras

# 数据增强函数
# 椒盐噪声
def SaltAndPepper(X, percetage):
    SP_NoiseImg = X.copy()
    SP_NoiseNum = int(percetage*X.shape[1]*X.shape[2])
    for i in range(X.shape[0]):
        for j in range(SP_NoiseNum):
            rand1 = np.random.randint(0, X.shape[1]-1)
            rand2 = np.random.randint(0, X.shape[2]-1)
            if np.random.randint(0, 1) == 0:

                SP_NoiseImg[i, rand1, rand2, 0] = 0
            else:
                SP_NoiseImg[i, rand1, rand2, 0] = 1
    oX = np.append(X, SP_NoiseImg, axis=0)
    return oX

# 高斯噪声
def addGaussianNoise(X, percetage):
    G_Noiseimg = X.copy()
    w = X.shape[1]
    h = X.shape[2]
    G_NoiseNum=int(percetage*w*h)
    for i in range(X.shape[0]):
        for j in range(G_NoiseNum):
            temp_x = np.random.randint(0, h)
            temp_y = np.random.randint(0, w)
            G_Noiseimg[i, temp_x, temp_y, 0] = np.random.randn(1)[0]

    oX = np.append(X, G_Noiseimg, axis=0)
    return oX

# 旋转
def rotate(X, angle, scale=1.0):
    rotated = X.copy()
    w = X.shape[1]
    h = X.shape[2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    for i in range(X.shape[0]):
        rotated[i, :, :, 0] = cv2.warpAffine(X[i, :, :, 0], m, (w, h))

    oX = np.append(X, rotated, axis=0)
    return oX

# 翻转
def flip(X):
    flipped_image = X.copy()
    for i in range(X.shape[0]):
        flipped_image[i] = np.fliplr(X[i])

    oX = np.append(X, flipped_image, axis=0)
    return oX