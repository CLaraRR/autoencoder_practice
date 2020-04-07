'''
降噪自编码器（DAE）
1、给原始图像加上随机噪声，噪声呈高斯分布
2、编解码器均为CNN
'''
from keras.layers import Input, Dense, UpSampling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import plot_model
import os
# 指定gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz2.38/bin/'

# 输入维度
input_img = Input(shape=(28, 28, 1))
# 基于卷积和池化的编码器
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)
# 基于卷积核上采样的解码器
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
# 搭建模型并编译
autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# 准备加了噪声的mnist数据
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data('mnist.npz')
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
# 给数据添加噪声
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# 对噪声数据进行自编码训练
autoencoder.fit(x_train_noisy, x_train,
                nb_epoch=10,
                batch_size=512,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# 保存模型
autoencoder.save('./model/model_dae_cnn')

# 模型画图
plot_model(autoencoder, to_file='model_dae_cnn.png', show_shapes=True)
model_img = mpimg.imread('model_dae_cnn.png')
plt.imshow(model_img)
plt.axis('off')
plt.show()

# 加载模型
new_model = load_model('./model/model_dae_cnn')
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


