'''
变分自编码器（VAE）：VAE不是将输入图像压缩伟潜在空间的编码，
而是将图像转换为最常见的两个统计分布参数——均值和标准差，
然后使用这两个参数来从分布中进行随机采样得到隐变量，
对隐变量进行解码重构即可。
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import to_categorical
import os

# 指定gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##### 设置模型相关参数 #####
batch_size = 256
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0
num_classes = 10

##### 加载mnist数据集 #####
(x_train, y_train_), (x_test, y_test_) = mnist.load_data('mnist.npz')
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = to_categorical(y_train_, num_classes)
y_test = to_categorical(y_test_, num_classes)

##### 建立计算均值和方差的编码网络 #####
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

##### 定义参数复现技巧函数和抽样层 #####
# 参数复现技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)),
        mean=0.,
        stddev=epsilon_std
    )
    return z_mean + epsilon*K.exp(z_log_var/2)
# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

##### 定义模型解码部分（生成器） #####
# 解码层
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

##### 接下来实例化三个模型 #####
# 1、一个端到端的自动编码器，用于完成输入信号的重构
vae = Model(x, x_decoded_mean)
# 2、一个用于将输入空间映射为隐空间的编码器
encoder = Model(x, z_mean)
# 3、一个利用隐空间的分布产生的样本点生成对应的重构样本的生成器
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

##### 定义VAE损失函数并进行训练 #####
# xent_loss是重构损失，kl_loss是KL loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
# add_loss是新增的方法，用于更灵活的添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

# 开始训练
vae.fit(
    x_train,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, None)
)
# 保存模型
vae.save('./model/model_vae')
encoder.save('./model/model_vae_encoder')
generator.save('./model/model_vae_generator')

##### 测试一下模型的重构效果 #####
decoded_imgs = vae.predict(x_test)
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # 展示原始图像
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
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



##### 测试模型的生成能力（从隐空间采样然后利用训练好的生成器生成） #####
# VAE是一个生成模型，可以用它来生成新数字
# 可以从隐平面上采样一些点，然后生成对应的显变量，即MNIST的数字
# 观察隐变量的两个维度变化是如何影响输出结果的
n = 15
# figure with 15*15 digits
digit_size = 28
figure = cnp.zeros((digit_size*n, digit_size*n))
# 用正态分布的分位数来构建隐变量对
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i*digit_size:(i+1)*digit_size, j*digit_size:(j+1)*digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()