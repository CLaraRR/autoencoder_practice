'''
稀疏自编码器(Sparse AutoEncoder)
使用KL散度对神经元稀疏化
'''
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.datasets import mnist
from keras import backend as K
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
# 指定gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##### 设置网络参数 #####
p = 0.05  # 使大部分神经元的激活值（所有神经元的激活值的平均值）接近这个p值
beta = 3  # 控制KL散度所占的比重
input_dim = 784
encoding_dim = 30
lambda_val = 0.001  # weight decay
epochs = 400
batch_size = 2048



#### 准备mnist数据 ######
(x_train, y_train_), (x_test, y_test_) = mnist.load_data('mnist.npz')
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

##### 定义网络 ######
input_img = Input(shape=(input_dim,))

# 自定义正则项函数, 计算KL散度
def sparse_reg(activity_matrix):
    activity_matrix = K.softmax(activity_matrix, axis=0)  # 把激活值先用softmax归一化
    p_hat = K.mean(activity_matrix, axis=0)  # 将第j个神经元在batch_size个输入下所有的输出激活值取平均
    print('p_hat=', p_hat)
    KLD = p*(K.log(p/p_hat))+(1-p)*(K.log((1-p)/(1-p_hat)))  # 计算KL散度
    print('KLD=', KLD)
    return beta*K.sum(KLD)  # 所有神经元的KL散度相加并乘以beta

encoded = Dense(
    encoding_dim,
    activation='relu',
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

sae.compile(optimizer='adam', loss='binary_crossentropy')
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