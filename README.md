# DARTS

This is a reimplementation of original DARTS ([darts](https://github.com/quark0/darts)), Architect class is inspired by newer implementation ([pt.darts](https://github.com/khanrc/pt.darts)).

This implementation brings DARTS algorithm to TensorFlow 2, allowing for further experiments with TensorFlow compatible layers. The aim is to provide a readable and customizable implementation.

This work is based not only on the implementation listed above, but the original paper as well:

>Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018). [[arxiv](https://arxiv.org/abs/1806.09055)]

## Requirements

* python 3.9
* tensorflow 2.8
* cudatoolkit 11.2
* cudnn 8.1
* tensorflow-addons 0.20

In order to use GPU, which is highly recommended, CUDA drivers have to be installed, which is very dependant on hardware itself and therefore won't be covered in installation part of this document. Convolution layers with approximate multipliers were only used on Barbora cluster, since these layers don't support more recent GPU architectures.

## Installation

The installation can be a bit problematic and requires the use of both anaconda and pip in order to install all required packages. The installation requires anaconda3 for environment management and package installation, if you don't have anaconda3 installed, see https://docs.anaconda.com/free/anaconda/install/linux/.

* Create an anaconda environment with required anaconda packages

```
$ conda create --name darts -c conda-forge cudatoolkit=11.2.2 cudnn=8.1 python=3.9
```

* Install additional packages with pip

```
$ pip install tensorflow==2.8 tensorflow-addons protobuf==3.20
```

In order to experiment with convolutions using approximate multipliers, you need to install these layers by following instructions in this [repository](https://github.com/ehw-fit/tf-approximate/tree/master/tf2).

## Usage

For cli arguments hints, see:
```
$ python3 train.py --help
```
and
```
$ python3 search_cell.py --help
```

### Architecture search

```
$ python3 search_cell.py
```

If out of memory error occurs, lower batch size by specifying cli argument `--batch_size x`, where `x` is number low enough that the model fits into memory (32, 16 or 4 possibly).
After each epoch a genotype is written to a file in `logs/search_arch/start_time_of_search/train/`, these genotypes can be used for evaluation.

### Architecture evaluation

```
$ python3 train.py --genotype_file logs/search_arch/20230409-231511/train/genotype_epoch_50 --batch_size 32 --epochs 600 --init_channels 42 --layers 20 --cutout --auxiliary --nodes 4 --multiplier 4
```

As in architecture search, adjust batch size to avoid OOM errors. Replace genotype file in order to experiment with architecture you found or use example file `reg_conv_gen` in root directory of this project.

## Experimenting with different operations

By default regular convolution operations are used. This project also provides operations using convolutions with approximate multipliers and attention operations. In order to use different operations, few changes have to be made to the code. Dynamic imports for new operations with cli arguments would be problematic and would require change of imports inside the code anyways, therefore to change operations used, changes are made only in the code.

In order to change the operations, follow there instructions:

* Replace `operations` import on lines 14 in `model_train.py` and `model_search.py` to your file with operations (There are commented out imports for approximate and attention operations in the code). Note that your operations file has to contain `ReLUConvBN` and `FactorizedReduce`, which should be the same as in any other operations set, because these operations are outside the search space in DARTS algorithm for preprocessing etc.

* Make sure `PRIMITIVES` in `genotypes.py` is a list with values same as operations inside operation dictionary, used in `operations.py`. These primitives are used to call operations from that dictionary and have to be the same.

* Convolutional layers with approximate multipliers use accurate multiplier (mul8u_1JFF.bin), in order to use different one, change `approx_mul_table_file` parameter in convolutional layer initialization.