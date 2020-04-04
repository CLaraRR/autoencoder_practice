'''
稀疏自编码器
'''
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
# 指定gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##### 设置网络参数 #####
sp = 0.01
n_val = 3  # control the activity of the hidden layer nodes
input_dim = 784
encoding_dim = 200
lambda_val = 0.001  # weight decay
# num_classes = 10
epochs = 400
batch_size = 2048



#### 准备mnist数据 ######
(x_train, y_train_), (x_test, y_test_) = mnist.load_data('mnist.npz')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# y_train = to_categorical(y_train_, num_classes)
# y_test = to_categorical(y_test_, num_classes)

##### 定义网络 ######
input_img = Input(shape=(input_dim,) )

# 自定义正则项函数
def sparse_reg(activity_matrix):
    p = 0.01
    beta = 3
    p_hat = K.mean(activity_matrix)  # average over the batch samples
    print('p_hat=', p_hat)
    KLD = p*(K.log(p/p_hat))+(1-p)*(K.log((1-p)/(1-p_hat)))
    print('KLD=', KLD)
    return beta*K.sum(KLD)  # sum over the layer units

encoded = Dense(
    encoding_dim,
    activation='sigmoid',
    kernel_regularizer=regularizers.l2(lambda_val/2),
    activity_regularizer=sparse_reg
)(input_img)

decoded = Dense(
    input_dim, activation='sigmoid',
    kernel_regularizer=regularizers.l2(lambda_val/2),
    activity_regularizer=sparse_reg
)(encoded)
# sae模型
sae = Model(input_img, decoded)


# encoder模型
encoder = Model(input_img, encoded)

# decoder模型
decoded_input = Input(shape=(encoding_dim,))
decoder_layer = sae.layers[-1](decoded_input)
decoder = Model(decoded_input, decoder_layer)

sae.compile(optimizer='sgd', loss='mse')
sae.summary()
# 开始训练
sae.fit(
    x_train,
    x_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# 用测试数据集看看重构的效果
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
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