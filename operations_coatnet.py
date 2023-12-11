"""
Module providing many operations used in preprocessing and in search space, Attention and
Feed-Forward operations are inspired by CoAtNet article: https://arxiv.org/abs/2106.04803.

This code is part of reimplementation of original DARTS
and is based on it, the original implementation
can be found here: https://github.com/quark0/darts
and is licensed under Apache 2.0

Author: Vojtech Eichler
Date: April 2023
"""

import tensorflow as tf
import tensorflow.keras as keras

OP_DICT = {
    'none': lambda C_curr, stride, normalize: Zero(stride),
    'avg_pool_3x3' : lambda C_curr, stride, normalize: AvgPool(3, stride=stride, normalize=normalize),
    'max_pool_3x3' : lambda C_curr, stride, normalize: MaxPool(3, stride=stride, normalize=normalize),
    'sep_conv_3x3': lambda C_curr, stride, normalize: SepConv(C_curr, 5, stride),
    'attention': lambda C_curr, stride, normalize: Attention(C_curr, stride),
    'ffn': lambda C_curr, stride, normalize: FeedForwardNet(C_curr, C_curr, stride),
    'skip_connect' : lambda C_curr, stride, normalize: Identity() if stride[0] == 1 else FactorizedReduce(C_curr),
}

class SepConv(keras.layers.Layer):
    """Separable convolution operation with ReLU activation
    at the beginning of the operation and batch normalization at the end

    """
    def __init__(self, C_curr, kernel_size, stride):
        super().__init__()
        self.relu = keras.layers.ReLU()
        self.sep_conv = keras.layers.SeparableConv2D(C_curr, kernel_size=kernel_size, strides=stride, padding='same')
        self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        x = self.relu(x)
        x = self.sep_conv(x)
        x = self.bn(x)
        return x

class Attention(keras.layers.Layer):
    """Self-attention operation which is optionally down-sampled by max pooling
    if stride is 2. This self-attention uses 2 heads

    """
    def __init__(self, C_curr, stride, head_dim=32, activation='gelu'):
        super().__init__()
        self.C_curr = C_curr
        self.stride = stride
        self.head_dim = head_dim
        self.activation = activation
        self.head_n = 2

        self.preact = keras.layers.LayerNormalization(epsilon=1e-5)
        self.max_pool_2 = keras.layers.MaxPool2D(pool_size=2, strides=self.stride, padding='same')
        self.multihead_attn = keras.layers.MultiHeadAttention(self.head_n, self.head_dim, output_shape=C_curr, use_bias=False)

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        preact = self.preact(x)
        if self.stride != 1:
            op = self.max_pool_2(preact)
        op = self.multihead_attn(op, op)
        return op

class FeedForwardNet(keras.layers.Layer):
    """Feed-forward operation using two dense layers and max pooling

    """
    def __init__(self, C_curr, C_out, stride):
        super().__init__()
        self.dense_1 = keras.layers.Dense(C_out, activation='relu')
        self.dense_2 = keras.layers.Dense(C_curr)
        self.maxpool = keras.layers.MaxPool2D(1, stride, padding='same')

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        op = self.dense_1(x)
        op = self.dense_2(op)
        return self.maxpool(op)

class Identity(keras.layers.Layer):
    """Shortcut operation for normal cells

    """
    def __init__(self):
        super().__init__()

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        return x

class Zero(keras.layers.Layer):
    """Special zero operation representing no connection

    """
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        return tf.zeros_like(x)[:, ::self.stride[0], ::self.stride[1], :]

class AvgPool(keras.layers.Layer):
    """Average pooling operation with batch normalization

    """
    def __init__(self, kernel_size, stride, normalize):
        super().__init__()
        self.normalize = normalize
        self.pool = keras.layers.AveragePooling2D(kernel_size, strides=stride, padding='same')
        if normalize:
            self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        x = self.pool(x)
        if self.normalize:
            x = self.bn(x)
        return x

class MaxPool(keras.layers.Layer):
    """Max pooling operation with batch normalization

    """
    def __init__(self, kernel_size, stride, normalize):
        super().__init__()
        self.normalize = normalize
        self.pool = keras.layers.MaxPooling2D(kernel_size, strides=stride, padding='same')
        if normalize:
            self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        x = self.pool(x)
        if self.normalize:
            x = self.bn(x)
        return x

class ReLUConvBN(keras.layers.Layer):
    """Operation which first applies ReLU activation function, then convolution
    and batch normalization

    """
    def __init__(self, C_out, kernel_size, stride, padding):
        super().__init__()
        self.relu = keras.layers.ReLU()
        self.conv = keras.layers.Conv2D(C_out, kernel_size, stride, padding, use_bias=False)
        self.bn = keras.layers.BatchNormalization(momentum=0.15)

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        op = self.relu(x)
        op = self.conv(op)
        return self.bn(op)

class FactorizedReduce(keras.layers.Layer):
    """Shortcut operation for reduction cells

    """
    def __init__(self, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = keras.layers.ReLU()
        self.conv_1 = keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.conv_2 = keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.bn = keras.layers.BatchNormalization(momentum=0.15)

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        x = self.relu(x)
        out = tf.concat([self.conv_1(x), self.conv_2(x[:,1:,1:,:])], -1)
        out = self.bn(out)
        return out
