"""
Module providing model for searching cells architecture

This code is part of reimplementation of original DARTS
and is based on it, the original implementation
can be found here: https://github.com/quark0/darts
and is licensed under Apache 2.0

Author: Vojtech Eichler
Date: April 2023
"""

from genotypes import PRIMITIVES, Genotype
from operations import * # Comment out this line if any of the next two imports is used

# For approx experiments uncomment next line:
#from operations_approx import *

# For CoAtNet experiments uncomment next line:
#from operations_coatnet import *

import tensorflow as tf
import tensorflow.keras as keras

class MixedOp(keras.layers.Layer):
    """Class for mixed operations between hidden states

    """
    def __init__(self, C_curr, stride, auxiliary_skip, auxiliary_op):
        """Mixed operation initialization method

        Args:
            C_curr : Number of channels for operations
            stride : strides
        """
        super().__init__()
        self.stride = stride
        self._ops = []

        if auxiliary_skip:
            if self.stride[0] == 2:
                self.aux_op = FactorizedReduce(C_curr)
            elif auxiliary_op == 'skip_connect':
                self.aux_op = Identity()

        # Initialize all operations from primitives
        for prim in PRIMITIVES:
            op = OP_DICT[prim](C_curr, stride, True)
            self._ops.append(op)

    def call(self, x, weights, aux_decay, training=None):
        weights = tf.cast(weights, tf.float16)
        weights = tf.reshape(weights, [len(PRIMITIVES), 1, 1, 1, 1])
        ops = [op(x, training) for op in self._ops]
        # Mix operations output, each operation is multiplied by corresponding weight
        res = tf.reduce_sum(ops * weights, axis=0)
        res += self.aux_op(x, training) * aux_decay
        return res

class Cell(keras.layers.Layer):
    """Class representing a cell which are stacked to form a network

    """
    def __init__(self, n_nodes, multiplier, C_curr, reduction, reduction_prev, auxiliary_skip, auxiliary_op):
        """Cell initialization method

        Args:
            n_nodes : Number of hidden states
            multiplier : Channel multiplier coefficient
            C_curr : Current channels
            reduction : Specify if this cell should be reduction cell
            reduction_prev : Specify if previous cell was reduction cell
        """
        super().__init__()
        self._reduction_prev = reduction_prev
        self.reduction = reduction
        self._n_nodes = n_nodes
        self._multiplier = multiplier
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_curr)
        else:
            self.preprocess0 = ReLUConvBN(C_curr, 1, 1, 'valid')
        self.preprocess1 = ReLUConvBN(C_curr, 1, 1, 'valid')

        # Initialize all mixed operations between hidden states
        self._ops = []
        for i in range(self._n_nodes):
            for j in range (i + 2):
                stride = [2,2] if reduction and j < 2 else [1,1]
                self._ops.append(MixedOp(C_curr, stride, auxiliary_skip, auxiliary_op))

    def call(self, s0, s1, weights, aux_decay, training=None):
        """Cell forward pass method

        Args:
            s0 : Output of second to last preceding cell or stem stage
            s1 : Output of last preceding cell or stem stage
            weights : Architecture weights
            training : Specify training or inference mode. Defaults to None

        Returns:
            returns concatenated hidden states
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._n_nodes):
            s = tf.math.add_n(self._ops[offset + j](h, weights[offset + j], aux_decay, training) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # Concatenate hidden states output
        return tf.concat(states[-self._multiplier:], -1)

class Network(keras.Model):
    """Class for model which is used for searching the search space

    """
    def __init__(self, C, criterion, n_classes, n_layers, n_nodes=4, multiplier=4, stem_multiplier=3, auxiliary_skip=False, auxiliary_op=None):
        """Network initialization method

        Args:
            C : Initial number of channels
            criterion : Loss function
            n_classes : Number of classification classes
            n_layers : Number of layers
            n_nodes : Number of hidden states in a cell. Defaults to 4.
            multiplier : Channel multiplier coefficient. Defaults to 4.
            stem_multiplier : Channel multiplier for stem stage. Defaults to 3.
        """
        super(Network, self).__init__()
        self._C = C
        self._n_classes = n_classes
        self._n_layers = n_layers
        self._n_nodes = n_nodes
        self._multiplier = multiplier
        self._criterion = criterion
        self._aux_decay = 0

        # Stem stage
        C_curr = C * stem_multiplier
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

            cell = Cell(n_nodes, multiplier, C_curr, reduction, reduction_prev, auxiliary_skip, auxiliary_op)
            reduction_prev = reduction
            self.cells.append(cell)

        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(n_classes, activation=None)

        self._initialize_alphas()

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Logits from network classifier
        """
        op = self.stem_1(x)
        op = self.stem_2(op)
        s0 = s1 = op
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = tf.nn.softmax(self.alphas_reduce, axis=-1)
            else:
                weights = tf.nn.softmax(self.alphas_normal, axis=-1)
            s0, s1 = s1, cell(s0, s1, weights, self._aux_decay, training)
        out = self.global_pooling(s1)
        logits = self.classifier(out)
        return logits

    def _loss(self, x, target, training=False):
        """Method to calculate model loss using loss function specified during initalization

        Args:
            x : Input images
            target : Expected annotations for input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Loss value
        """
        logits = self(x, training=training)
        return self._criterion(target, logits)

    def _initialize_alphas(self):
        """Method which initializes architecture weights (alphas)
        """
        # Calculate shape for weights according to number of nodes and operations
        # in search space
        k = sum(1 for i in range(self._n_nodes) for n in range(i + 2))
        n_ops = len(PRIMITIVES)

        self.alphas_normal = tf.Variable(1e-3*tf.random.uniform([k, n_ops]), trainable=False)
        self.alphas_reduce = tf.Variable(1e-3*tf.random.uniform([k, n_ops]), trainable=False)
        self._arch_params = [self.alphas_normal, self.alphas_reduce]

    def arch_params(self):
        """Getter method for architecture weights

        Returns:
            architecture weights
        """
        return self._arch_params

    def _parse_alphas(self, weights):
        """Method to parse architecture weights into genes

        Args:
            weights : Architecture weights (alphas)

        Returns:
            Either normal or reduction gene
        """
        weights = tf.constant(weights)
        gene = []
        n = 2
        start = 0
        for i in range(self._n_nodes):
            end = start + n
            W = weights[start:end]
            edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            node_gene = []
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'): # Don't include zero operation
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                node_gene.append((PRIMITIVES[k_best], j))
            gene.append(node_gene)
            start = end
            n += 1
        return gene

    def genotypes(self):
        """Method which creates a genotype out of architecture weights

        Returns:
            Parsed genotype
        """
        gene_normal = self._parse_alphas(tf.nn.softmax(self.alphas_normal))
        gene_reduce = self._parse_alphas(tf.nn.softmax(self.alphas_reduce))

        concat = range(2 + self._n_nodes - self._multiplier, self._n_nodes + 2)
        return Genotype(gene_normal, concat, gene_reduce, concat)