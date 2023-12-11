"""
Script for searching cell architectures

This code is part of reimplementation of original DARTS
and is based on it, the original implementation
can be found here: https://github.com/quark0/darts
and is licensed under Apache 2.0

Author: Vojtech Eichler
Date: April 2023
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from model_search import *
from config import Config
from architect import Architect
import datetime

LOG_DIR='./logs'

tf.get_logger().setLevel('INFO')
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

def linear_decay(epoch):
    return 1.0 * ((50 - epoch) / 50)

@tf.function
def validation_step(x_batch_valid, y_batch_valid):
    logits = model(x_batch_valid, training=False)
    loss = criterion(y_batch_valid, logits)
    validation_acc.update_state(y_batch_valid, logits)
    valid_loss.update_state(y_batch_valid, logits)
    return loss

@tf.function
def train_step(x_batch_train, y_batch_train):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        loss = criterion(y_batch_train, logits)

    grads = tape.gradient(loss, model.trainable_weights)
    # Apply gradient clipping
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=config.args.grad_clip)
    # Apply weight decay
    for var in model.trainable_weights:
        var.assign_sub(var * config.args.weight_decay * lr)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_loss.update_state(y_batch_train, logits)
    train_acc.update_state(y_batch_train, logits)
    return loss

@tf.function
def architect_step(x_batch_train, y_batch_train, x_batch_valid, y_batch_valid):
    architect.step(x_batch_train, y_batch_train, x_batch_valid, y_batch_valid, xi=lr, net_optimizer=optimizer, unrolled=config.args.unrolled)

# This is function taken directly from keras implementation, since current
# learning rate is needed and in tf-2.8 it's not possible to get it from
# optimizer object nor from learning rate scheduler, the implementation is
# taken from: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
def current_lr(step, decay_steps, alpha, initial_lr):
    step = min(step + 1, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_lr * decayed

# Data augmentation transformations
def trans(x_train, y_train):
    x_train = tf.image.resize_with_pad(x_train, 40, 40)
    x_train = keras.layers.RandomCrop(32, 32)(x_train)
    x_train = keras.layers.RandomFlip("horizontal")(x_train)

    return x_train, y_train

config = Config('search')
tf.random.set_seed(config.args.seed)

(x, y), (x_, y_) = keras.datasets.cifar10.load_data()

# Normalize data
x = x / 255

# Prepare dataset with data augmentation
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=50000)
train_dataset = dataset.take(25000)
val_dataset = dataset.skip(25000).take(25000)
train_dataset = train_dataset.shuffle(buffer_size=25000).batch(config.args.batch_size)
train_dataset = train_dataset.map(lambda x1, y1: trans(x1, y1))
val_dataset = val_dataset.shuffle(buffer_size=25000).batch(config.args.batch_size)
val_dataset = val_dataset.map(lambda x1, y1: trans(x1,y1))

# Calculate number of steps for learning rate decay
decay_steps = config.args.epochs * len(train_dataset)

# Initialize learing rate scheduler, loss function and optimizer
lr_scheduler = keras.experimental.CosineDecay(config.args.learning_rate, decay_steps, config.args.learning_rate_min)
criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=config.args.momentum)

# Create a model and an architect
model = Network(config.args.init_channels, criterion, 10, config.args.layers, n_nodes=config.args.nodes, multiplier=config.args.multiplier,
                auxiliary_skip=True, auxiliary_op='skip_connect')
architect = Architect(model, config.args, criterion)

# Setup tensorboard
tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
tb_callback.set_model(model)

lr = tf.cast(config.args.learning_rate, tf.float32)
lr_step = 0

# Initialize metrics
train_loss = keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
train_acc = keras.metrics.SparseCategoricalAccuracy()
valid_loss = keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
validation_acc = keras.metrics.SparseCategoricalAccuracy()
best_acc = 0
best_genotype = model.genotypes()

# prepare log directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/search_arch/' + current_time + '/train'
test_log_dir = 'logs/search_arch/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


with open(f"{train_log_dir}/genotype_initial", 'w') as genotype_file:
    genotype_file.write(str(best_genotype))
with open(f"{train_log_dir}/config", 'w') as config_file:
    config_file.write(str(config.args))
print(f"Initial genotype: {best_genotype}")
print(f"Initial alphas: {tf.nn.softmax(model.arch_params(), axis=-1)}")

for epoch in range(config.args.epochs):
    #model._aux_decay = architect.v_model._aux_decay = linear_decay(epoch)
    # Training
    for step, ((x_batch_train, y_batch_train), (x_batch_valid, y_batch_valid)) in enumerate(zip(train_dataset, val_dataset)):
        # First build the model
        #if epoch == 0 and step == 0:
            #architect.v_model._loss(x_batch_valid, y_batch_valid)

        lr = tf.cast(current_lr(lr_step, decay_steps, config.args.learning_rate_min, config.args.learning_rate), tf.float32)
        architect_step(x_batch_train, y_batch_train, x_batch_valid, y_batch_valid)
        loss = train_step(x_batch_train, y_batch_train)
        lr_step += 1

        if (step + 1) % 100 == 0:
            print(datetime.datetime.now())
            print(f'Epoch: {epoch + 1}')
            print(f'Step: {step + 1}')
            print(f'Number of samples seen: {(step + 1) * config.args.batch_size}')
            print(f"Loss is: {loss}")
            print(f"Learning rate: {lr}\n")


    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

    trn_acc = train_acc.result()

    # Validation
    for step, (x_batch_valid, y_batch_valid) in enumerate(val_dataset):
        loss = validation_step(x_batch_valid, y_batch_valid)
        if (step + 1) % 100 == 0:
            print(datetime.datetime.now())
            print(f'Epoch: {epoch + 1}')
            print(f'Validation step: {step + 1}')
            print(f"Validation loss is: {loss}\n")

    val_acc = validation_acc.result()

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', valid_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', validation_acc.result(), step=epoch)

    # End of epoch logging and updating/reseting metrics
    with open(f"{train_log_dir}/genotype_epoch_{epoch + 1}", 'w') as genotype_file:
        genotype_file.write(str(model.genotypes()))

    if (val_acc > best_acc or epoch == 0):
        best_acc = val_acc
        best_genotype = model.genotypes()

    print(f"End of epoch {epoch + 1}")
    print(f"Validation accuracy: {float(val_acc)}")
    print(f"Genotype: {model.genotypes()}")
    print(f"Alphas: {tf.nn.softmax(model.arch_params(), axis=-1)}\n\n")

    train_loss.reset_states()
    valid_loss.reset_states()
    train_acc.reset_states()
    validation_acc.reset_states()

# End of architecture search
print(f"Best accuracy is: {best_acc}")
print(f"This was achieved with this genotype: {best_genotype}")
print(f"Alphas: {tf.nn.softmax(model.arch_params(), axis=-1)}")
model.summary()
with open(f"{train_log_dir}/genotype_best", 'w') as genotype_file:
    genotype_file.write(str(best_genotype))