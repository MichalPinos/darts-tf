"""
Module providing model for evaluating cells architecture

This code is part of reimplementation of original DARTS
and is based on it, the original implementation
can be found here: https://github.com/quark0/darts
and is licensed under Apache 2.0

Author: Vojtech Eichler
Date: April 2023
"""

from genotypes import PRIMITIVES, Genotype
from operations import *
# For approx experiments uncomment next line:
#from operations_approx import *
# For CoAtNet experiments uncomment next line:
#from operations_coatnet import *
import tensorflow as tf
import tensorflow.keras as keras
from config import base_config as config


class AuxiliaryHeadCifar(keras.layers.Layer):
    """Auxiliary classifier class
    """
    def __init__(self, n_classes):
        super().__init__()
        self.relu1 = keras.layers.ReLU()
        self.avg_pool = keras.layers.AveragePooling2D(5, strides=3, padding='valid')
        self.conv1 = keras.layers.Conv2D(128, kernel_size=1, use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.relu2 = keras.layers.ReLU()
        self.conv2 = keras.layers.Conv2D(768, 2, use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.relu3 = keras.layers.ReLU()
        self.flatten = keras.layers.Flatten()
        self.criterion = keras.layers.Dense(n_classes)

    def call(self, x, training=None):
        """Forward pass method for auxiliary classifier

        Args:
            x : input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            auxiliary loss
        """
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.criterion(x)
        return x

class Cell(keras.layers.Layer):
    """Class representing a cell which are stacked to form a model

    """
    def __init__(self, n_nodes, C_curr, reduction, reduction_prev, genotype, drop_rate):
        """Cell initialization method

        Args:
            n_nodes : Number of hidden states
            C_curr : Current number of channels
            reduction : Specify if this cell should be reduction cell
            reduction_prev : Specify if previous cell was reduction cell
            genotype : Architecture genotype string
            drop_rate : drop rate for dropout layers
        """
        super().__init__()
        self._reduction_prev = reduction_prev
        self.reduction = reduction
        self._n_nodes = n_nodes
        self.dropout = tf.keras.layers.Dropout(drop_rate)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_curr)
        else:
            self.preprocess0 = ReLUConvBN(C_curr, 1, 1, 'valid')
        self.preprocess1 = ReLUConvBN(C_curr, 1, 1, 'valid')

        # Initialize operations according to genotype
        gene = genotype.reduce if reduction else genotype.normal
        self.concat = genotype.reduce_concat if reduction else genotype.normal_concat
        self.ops = []
        self.indices = []
        for edges in gene:
            for op in edges:
                stride = [2, 2] if reduction and op[1] < 2 else [1, 1]
                self.ops.append(OP_DICT[op[0]](C_curr, stride, False))
                self.indices.append(op[1])
                
    def drop_connection(self, x, is_training, survival_prob):
        """Drop the entire conv with given survival probability."""
        # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
        # Implementation taken from EfficientNet, automl
        
        if not is_training:
            return x

        # Compute tensor.
        batch_size = tf.shape(x)[0]
        random_tensor = survival_prob
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        # Unlike conventional way that multiply survival_prob at test time, here we
        # divide survival_prob at training time, such that no addition compute is
        # needed at test time.
        output = x / survival_prob * binary_tensor
        return output                

    def residual(self, inputs0, inputs1, x, is_training):
        
        if config["survival_prob"]:
            x = self.drop_connection(x, is_training, config["survival_prob"])
        x = tf.add_n([inputs0, x])
        return x
            

    def call(self, s0, s1, training=None):
        """Forward pass method of cell

        Args:
            s0 : Output of second to last preceding cell or stem stage
            s1 : Output of last preceding cell or stem stage
            training : Specify training or inference mode

        Returns:
            returns concatenated hidden states
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._n_nodes):
            s_cur = 0
            for j in range(2):
                s_tmp = self.ops[i * 2 + j](states[self.indices[i * 2 +j]], training)
                # Apply dropout to operations other than shortcut
                # if not isinstance(self.ops[i * 2 + j], Identity):  
                #     s_cur += self.dropout(s_tmp, training=training)
                # else:
                s_cur += s_tmp
            states.append(s_cur)
        out = tf.concat([states[i] for i in self.concat], -1)    
        return out

class Network(keras.Model):
    """Class for model which is used for evaluating architectures

    """
    def __init__(self, C, criterion, n_classes, n_layers, genotype, drop_rate, n_nodes=4, multiplier=4, stem_multiplier=3, auxiliary=False):
        """Network initialization method

        Args:
            C : Initial number of channels
            criterion : Loss function
            n_classes : Number of classification classes
            n_layers : Number of layers
            genotype : Architecture genotype string
            drop_rate : Drop probability for dropout layers
            n_nodes : Number of hidden states in each cell. Defaults to 4.
            multiplier : Channel multiplier coefficient. Defaults to 4.
            stem_multiplier : Channel multiplier for stem stage. Defaults to 3.
            auxiliary : Specify if auxiliary classifier should be used. Defaults to False.
        """
        super().__init__()
        self._C = C
        self._n_classes = n_classes
        self._n_layers = n_layers
        self._n_nodes = n_nodes
        self._multiplier = multiplier
        self._criterion = criterion
        self._drop_rate = drop_rate
        self._auxiliary = auxiliary

        C_curr = C * stem_multiplier
        # Stem stage
        self.stem_1 = keras.layers.Conv2D(C_curr, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)
        self.stem_2 = keras.layers.BatchNormalization()

        C_curr = C

        # Initialize cells
        self.cells = []
        reduction_prev = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(n_nodes, C_curr, reduction, reduction_prev, genotype, drop_rate)
            reduction_prev = reduction
            self.cells.append(cell)
            
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(n_classes, activation=None)

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCifar(n_classes)

    def call(self, x, training=None):
        """Forward pass method of a network for architecture evaluation

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Logits from network classifier and auxiliary classifier
        """
        logits_aux = None
        op = self.stem_1(x)
        op = self.stem_2(op)
        s0 = s1 = op
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, training=training)
            if i == (2 * self._n_layers // 3):
                if self._auxiliary:
                    logits_aux = self.auxiliary_head(s1, training=training)

        # s1 = self.head_act(self.head_bn(self.head(s1)))
        out = self.global_pooling(s1)
        logits = self.classifier(out)
        return logits, logits_aux
    
    def summary(self):
        x = tf.keras.Input(shape=(config["train_image_size"], config["train_image_size"], 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=True))
        return model.summary()

    # def _loss(self, x, target):
    #     """Method to calculate model loss using loss function specified during initialization

    #     Args:
    #         x : Input images
    #         target : Expected annotations for input images

    #     Returns:
    #         Loss value
    #     """
    #     logits = self(x, training=True)
    #     return self._criterion(target, logits)