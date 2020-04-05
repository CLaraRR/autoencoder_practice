'''
堆叠降噪自编码器(Stacked Denoising AutoEncoder)
编解码各2层，且维数基于中心对称
各层维数：784->256->64->256->784
逐个预训练4个autoencoder：使用前一个训练好的ae的encoder输出作为下一个ae的输入来训练当前ae
堆叠ae:取前面预训练好的4个ae的encoder层前后连接起来，形成最终stacked ae的模型结构
堆叠ae的训练：使用预训练好的参数初始化stacked ae，然后进行全局训练优化
'''
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
# 指定gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##### 设置网络参数 #####
epochs = 400
batch_size = 2048
origin_dim = 784
h_dim1 = 256
h_dim2 = 64


##### 准备mnist数据 ######
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

##### 构建单个autoencoder #####
class AutoEncoderLayer():
    def __init__(self, input_dim, output_dim, trainable=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.trainable = trainable
        self.build()

    def build(self):
        self.input = Input(shape=(self.input_dim,))
        self.encode_layer = Dense(self.output_dim, activation='relu')
        self.encoded = self.encode_layer(self.input)
        self.encoder = Model(self.input, self.encoded)

        self.decode_layer = Dense(self.input_dim, activation='sigmoid')
        self.decoded = self.decode_layer(self.encoded)

        self.autoencoder = Model(self.input, self.decoded)


    def set_trainable(self, trainable):
        self.trainable = trainable
        self.autoencoder.trainable = trainable
        self.encoder.trainable = trainable


class StackedAutoEncoder():
    def __init__(self, layer_list):
        self.layer_list = layer_list
        self.build()

    def build(self):
        out = self.layer_list[0].encoded
        for i in range(1, num_layers):
            out = self.layer_list[i].encode_layer(out)
        self.model = Model(self.layer_list[0].input, out)




def train_layers(encoder_list=None, layer=None, epochs=None, batch_size=None):
    '''
    预训练：逐层训练，当训练第l层时，将前（l-1）层的参数冻结
    :param encoder_list:
    :param layer:
    :param epochs:
    :param batch_size:
    :return:
    '''
    # 冻结当前层之前的所有层的参数,ps:第0层没有前置层
    # 对前(layer-1)冻结了的层用已经训练好的参数进行前向计算
    out = x_train_noisy
    if layer != 0:
        for i in range(layer):
            # print("encoder weight", str(i), ":", encoder_list[i].encoder.get_weights()[0])
            out = encoder_list[i].encoder.predict(out)

    encoder_list[layer].autoencoder.summary()
    encoder_list[layer].autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    encoder_list[layer].autoencoder.fit(
        out,
        out,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0
    )


def train_whole(sae=None, epochs=None, batch_size=None):
    '''
    用预训练好的参数初始化stacked ae的参数，然后进行全局训练优化
    :param model:
    :param epochs:
    :param batch_size:
    :return:
    '''
    # print("stacked sae weights:")
    # print(sae.model.get_weights())
    sae.model.summary()
    sae.model.compile(optimizer='adadelta', loss='binary_crossentropy')
    sae.model.fit(
        x_train_noisy,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
        verbose=2
    )


# 实例化4个ae
num_layers = 4
encoder_1 = AutoEncoderLayer(origin_dim, h_dim1, trainable=True)
encoder_2 = AutoEncoderLayer(h_dim1, h_dim2, trainable=True)
decoder_3 = AutoEncoderLayer(h_dim2, h_dim1, trainable=True)
decoder_4 = AutoEncoderLayer(h_dim1, origin_dim, trainable=True)
autoencoder_list = [encoder_1, encoder_2, decoder_3, decoder_4]

# 按照顺序对每一层进行预训练
print("Pre training:")
for level in range(num_layers):
    print("level:", level)
    train_layers(encoder_list=autoencoder_list, layer=level, epochs=epochs, batch_size=batch_size)


stacked_ae = StackedAutoEncoder(autoencoder_list)
print("Whole training:")
train_whole(sae=stacked_ae, epochs=epochs, batch_size=batch_size)


decoded_imgs = stacked_ae.model.predict(x_test_noisy)
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