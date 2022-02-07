import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 选择哪一块gpu,如果是-1，就是调用cpu
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import tensorflow as tf
import tensorflow.keras as keras
from img_aug import SaltAndPepper, addGaussianNoise, rotate, flip

# 运行Tensorflow前的准备工作
tf.compat.v1.disable_eager_execution()  # 关闭先前的会话
# 设置显存按需分配
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# 参数设置
input_size = 64
dropout_p = 0.3
lamda = 0.001
# 加载数据集
objectdict = {
    "2S1": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "BMP2(SN_9566)": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    "BRDM_2": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    "BTR70(SN-C71)": np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    "BTR_60": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    "D7": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    "T62": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    "T72(SN_132)": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    "ZIL131": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    "ZSU_23_4": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
}

train_path = np.loadtxt(open("train.csv", "rb"), dtype=str, delimiter=",", skiprows=1, usecols=[0])
test_path = np.loadtxt(open("test.csv", "rb"), dtype=str, delimiter=",", skiprows=1, usecols=[0])
train_name = np.loadtxt(open("train.csv", "rb"), dtype=str, delimiter=",", skiprows=1, usecols=[1])
test_name = np.loadtxt(open("test.csv", "rb"), dtype=str, delimiter=",", skiprows=1, usecols=[1])

train_X = np.zeros(shape=[len(train_path), input_size, input_size, 1])
train_Y = np.zeros(shape=[len(train_path), 10])

for i in range(len(train_path)):
    image = np.array(plt.imread(train_path[i]))
    image = resize(image, output_shape=(input_size, input_size, 1))
    train_X[i] = image
    train_Y[i] = objectdict[train_name[i]]

# train_X = SaltAndPepper(train_X, 0.1)
# train_X = addGaussianNoise(train_X, 0.2)
train_X = rotate(train_X, 20, scale=1.0)
# train_X = flip(train_X)
train_Y = np.append(train_Y, train_Y, axis=0)

# 打乱训练集的顺序
m = train_X.shape[0]
permutation = list(np.random.permutation(m))
train_X = train_X[permutation]
train_Y = train_Y[permutation]

test_X = np.zeros(shape=[len(test_path), input_size, input_size, 1])
test_Y = np.zeros(shape=[len(test_path), 10])


for i in range(len(test_path)):
    image = np.array(plt.imread(test_path[i]))
    image = resize(image, output_shape=(input_size, input_size, 1))
    test_X[i] = image
    test_Y[i] = objectdict[test_name[i]]

# 搭建模型
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=5, activation='relu', input_shape=(input_size, input_size, 1)))
model.add(keras.layers.MaxPool2D((2, 2), strides=2))

model.add(keras.layers.Conv2D(64, kernel_size=5, activation='relu'))
model.add(keras.layers.MaxPool2D((2, 2), strides=2))

model.add(keras.layers.Conv2D(64, kernel_size=5, activation='relu'))
model.add(keras.layers.MaxPool2D((2, 2), strides=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu', activity_regularizer=keras.regularizers.l2(lamda)))
tf.keras.layers.Dropout(dropout_p)
model.add(keras.layers.Dense(10, activation='softmax', activity_regularizer=keras.regularizers.l2(lamda)))
tf.keras.layers.Dropout(dropout_p)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

# model.fit(train_X, train_Y, epochs=1,  batch_size=32)
model.fit(train_X, train_Y, epochs=30, batch_size=64)

y_predict = model.predict(test_X)

cla = np.argmax(y_predict, axis=1)
# print(test_Y[0], y_predict[0])

loss, accuracy = model.evaluate(test_X, test_Y)
print(loss, accuracy)