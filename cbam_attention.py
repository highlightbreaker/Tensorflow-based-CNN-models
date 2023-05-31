# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------


import tensorflow as tf
import tensorflow.contrib.slim as slim

import lib.config.config as cfg
from lib.nets.network import Network
from tensorflow.python.keras import layers, Model
# （1）通道注意力
def channel_attenstion(inputs, ratio=0.25):
    '''ratio代表第一个全连接层下降通道数的倍数'''
    channel = inputs.shape[-1]  # 获取输入特征图的通道数
    channel = int(channel)
    # 分别对输出特征图进行全局最大池化和全局平均池化
    # [h,w,c]==>[None,c]
    x_max = layers.GlobalMaxPooling2D()(inputs)
    x_avg = layers.GlobalAveragePooling2D()(inputs)

    # [None,c]==>[1,1,c]
    x_max = layers.Reshape([1, 1, channel])(x_max)  # -1代表自动寻找通道维度的大小
    x_avg = layers.Reshape([1, 1, channel])(x_avg)  # 也可以用变量channel代替-1

    # 第一个全连接层通道数下降1/4, [1,1,c]==>[1,1,c//4]
    dense1 = layers.Dense(int(channel * ratio), activation='relu')
    x_max = dense1(x_max)
    x_avg = dense1(x_avg)

    # # relu激活函数
    # x_max = layers.Activation('relu')(x_max)
    # x_avg = layers.Activation('relu')(x_avg)

    # 第二个全连接层上升通道数, [1,1,c//4]==>[1,1,c]
    dense2 = layers.Dense(int(channel), activation='relu')
    x_max = dense2(x_max)
    x_avg = dense2(x_avg)

    # 结果在相叠加 [1,1,c]+[1,1,c]==>[1,1,c]
    x = layers.Add()([x_max, x_avg])

    # 经过sigmoid归一化权重
    x = tf.nn.sigmoid(x)

    # 输入特征图和权重向量相乘，给每个通道赋予权重
    x = layers.Multiply()([inputs, x])  # [h,w,c]*[1,1,c]==>[h,w,c]

    return x


# （2）空间注意力机制
def spatial_attention(inputs):
    # 在通道维度上做最大池化和平均池化[b,h,w,c]==>[b,h,w,1]
    # keepdims=Fale那么[b,h,w,c]==>[b,h,w]
    x_max = tf.reduce_max(inputs, axis=3, keepdims=True)  # 在通道维度求最大值
    x_avg = tf.reduce_mean(inputs, axis=3, keepdims=True)  # axis也可以为-1

    # 在通道维度上堆叠[b,h,w,2]
    x = layers.concatenate([x_max, x_avg])

    # 1*1卷积调整通道[b,h,w,1]
    x = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)

    # sigmoid函数权重归一化
    x = tf.nn.sigmoid(x)

    # 输入特征图和权重相乘
    x = layers.Multiply()([inputs, x])

    return x

# （3）CBAM注意力
def CBAM_attention(inputs):
    # 先经过通道注意力再经过空间注意力
    x = channel_attenstion(inputs)
    x = spatial_attention(x)
    return x