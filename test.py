import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"# 选择哪一块gpu,如果是-1，就是调用cpu
import os.path
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras

tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(sess)

face_data = datasets.fetch_olivetti_faces()

i = 0
plt.figure(figsize=(20, 20))
for img in face_data.images:
    plt.subplot(20, 20, i+1)
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(face_data.target[i])
    i = i + 1
plt.show()

X = face_data.images
y = face_data.target

X = X.reshape(400, 64, 64, 1)
X = resize(X, output_shape=(400, 32, 32, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = keras.Sequential()

model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(32, 32, 1)))
model.add(keras.layers.MaxPool2D((2, 2), strides=2))

model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
model.add(keras.layers.MaxPool2D((2, 2), strides=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(40, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200)

y_predict = model.predict(X_test)

print(y_test[0], np.argmax(y_predict[0]))