"""
Module providing lists of operations and Genotype type

This code is part of reimplementation of original DARTS
and is based on it, the original implementation
can be found here: https://github.com/quark0/darts
and is licensed under Apache 2.0

Author: Vojtech Eichler
Date: April 2023
"""

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES_CONV = [
    'none',
    'avg_pool_3x3',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

COATNET_PRIMITIVES = [
    'sep_conv_3x3',
    'attention',
    'ffn',
    'none',
    'skip_connect',
    'avg_pool_3x3',
    'max_pool_3x3'
]

# Uncomment this line to use CoAtNet like operations
#PRIMITIVES = COATNET_PRIMITIVES

# Primitives for regular and approximate convolutions
PRIMITIVES = PRIMITIVES_CONV