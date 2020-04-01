'''
VAE,CNN版本

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Dense, Input
from keras.layers import Convolution2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model, load_model
from keras import backend as K
from keras.datasets import mnist
import os

# 指定gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##### 加载mnist数据集 #####
(x_train, y_train_), (x_test, y_test_) = mnist.load_data('mnist.npz')
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

##### 设置网络参数 #####
input_shape = (image_size, image_size, 1)
batch_size = 256
kernel_size = 3
filters = 16
latent_dim = 2  # 隐变量取2维是为了方便后面画图
epochs = 30

##### 建立计算均值和方差的编码网络 #####
x_in = Input(shape=input_shape)
x = x_in
for i in range(2):
    filters *=2
    x = Convolution2D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        strides=2,
        padding='same',
    )(x)

# 备份当前shape，等下构建decoder的时候要用
shape = K.int_shape(x)

x = Flatten()(x)  # 把多维的数据变为一维
x = Dense(16, activation='relu')(x)
# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

##### 定义参数复现技巧函数和抽样层 #####
# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + epsilon * K.exp(z_log_var/2)
# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

##### 定义模型解码部分（生成器） #####
# 先搭建一个独立的模型，再调用模型
latent_inputs = Input(shape=(latent_dim,))
x = Dense(shape[1]*shape[2]*shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)  # 把数据变回原来的维数

for i in range(2):
    x = Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        strides=2,
        padding='same'
    )(x)
    filters //=2

outputs = Conv2DTranspose(
    filters=1,
    kernel_size=kernel_size,
    activation='sigmoid',
    padding='same'
)(x)

# 搭建为一个独立的encoder
encoder = Model(x_in, z_mean)
# 独立的decoder
decoder = Model(latent_inputs, outputs)
x_out = decoder(z)
# 完整的vae
vae = Model(x_in, x_out)

# xent_loss是重构loss, kl_loss是KL loss
xent_loss = K.sum(K.binary_crossentropy(x_in, x_out), axis=[1, 2, 3])
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)  # 往模型加入自定义的loss

vae.compile(optimizer='rmsprop')
vae.summary()

# 开始训练
vae.fit(
    x_train,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, None)
)
vae.save('./model/model_vae_cnn')
encoder.save('./model/model_vae_cnn_encoder')
decoder.save('./model/model_vae_cnn_decoder')

# 观察各个数字在隐空间的分布
encoder_model = load_model('./model/model_vae_cnn_encoder', custom_objects={'sampling':sampling})
x_test_encoded = encoder_model.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6,6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
plt.colorbar()
plt.show()

# 观察隐变量的两个维度变化是如何影响输出结果的
decoder_model = load_model('./model/model_vae_generator', custom_objects={'sampling':sampling})
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# 用正态分布的分位数来构建隐变量对
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder_model.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()













