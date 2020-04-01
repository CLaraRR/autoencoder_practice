'''
降噪自编码器（DAE）
1、给原始图像加上随机噪声，噪声呈高斯分布
2、编解码器均为一层全连接层
'''
from keras.layers import Input, Dense
from keras.models import Model, load_model
import os

# 指定gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import warnings
warnings.filterwarnings('ignore')

##### 完整的自编码器模型构建 #####
# 编码潜在空间表征维度
encoding_dim = 32
# 自编码输入
input_img = Input(shape=(784,))
# 使用一个全连接网络来搭建编码器
encoded = Dense(encoding_dim, activation='relu')(input_img)
# 使用一个全连接网络来对编码器进行解码
decoded = Dense(784, activation='sigmoid')(encoded)
# 构建keras模型
autoencoder = Model(input=input_img, output=decoded)

##### 也可以把编码器和解码器当做单独的模型来使用 #####
# 编码器模型
encoder = Model(input=input_img, output=encoded)
# 解码器模型
encoded_input = Input(shape=(encoding_dim,))
decoded_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoded_layer(encoded_input))

##### 对自编码器模型进行编译并使用mnist数据集进行训练 #####
# 编译模型
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# 准备mnist数据
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data(path='mnist.npz')
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# 给数据添加噪声
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# 训练
autoencoder.fit(x_train_noisy, x_train, nb_epoch=50, batch_size=256, shuffle=True, validation_data=(x_test_noisy, x_test))
# 保存模型
autoencoder.save('./model/model_dae')

new_model = load_model('./model/model_dae')
##### 对原始输入图像和自编码器训练后重构的图像进行可视化 #####
import matplotlib.pyplot as plt

decoded_imgs = new_model.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # 展示原始图像
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 展示自编码器重构后的图像
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()