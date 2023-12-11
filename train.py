"""
Script for evaluating cell architectures

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
from model_train import *
from config import Config
import datetime

LOG_DIR='./logs'

tf.get_logger().setLevel('INFO')
config = Config('train')
tf.random.set_seed(config.args.seed)

@tf.function
def validation_step(x_batch_valid, y_batch_valid):
    logits, _ = model(x_batch_valid, training=False)
    loss = criterion(y_batch_valid, logits)
    validation_acc.update_state(y_batch_valid, logits)
    valid_loss.update_state(y_batch_valid, logits)
    return loss

@tf.function
def train_step(x_batch_train, y_batch_train):
    with tf.GradientTape() as tape:
        logits, logits_aux = model(x_batch_train, training=True) # maybe use training=True?
        loss = criterion(y_batch_train, logits)
        if config.args.auxiliary:
            loss_aux = criterion(y_batch_train, logits_aux)
            loss += config.args.auxiliary_weight * loss_aux
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
def trans(x, y):
    x = tf.image.resize_with_pad(x, 40, 40)
    x = keras.layers.RandomCrop(32, 32)(x)
    x = keras.layers.RandomFlip("horizontal")(x)
    x = tfa.image.random_cutout(x, (16, 16), 0)

    return x, y

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize data
x_train = x_train / 255
x_test = x_test / 255

# Prepare dataset with data augmentation
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=50000).batch(config.args.batch_size)
train_dataset = train_dataset.map(lambda x, y: trans(x, y))
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.shuffle(buffer_size=10000).batch(config.args.batch_size)

# Calculate number of steps for learning rate decay
decay_steps = config.args.epochs * len(x_train) // config.args.batch_size

# Initialize learing rate scheduler, loss function and optimizer
lr_scheduler = keras.experimental.CosineDecay(config.args.learning_rate, decay_steps, 0)
criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=config.args.momentum)

with open(config.args.genotype_file, "r") as f:
    genotype = f.read()

# Create a model
model = Network(config.args.init_channels, criterion, 10, config.args.layers, n_nodes=config.args.nodes, multiplier=config.args.multiplier, genotype=eval(genotype), drop_rate=config.args.drop_rate, auxiliary=config.args.auxiliary)

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

# Prepare log directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/train_arch/' + current_time + '/train'
test_log_dir = 'logs/train_arch/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# Save config inside a file
with open(f"{train_log_dir}/config", 'w') as config_file:
    config_file.write(str(config.args))

for epoch in range(config.args.epochs):
    # Training
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        lr = tf.cast(current_lr(lr_step, decay_steps, config.args.learning_rate_min, config.args.learning_rate), tf.float32)
        loss = train_step(x_batch_train, y_batch_train)
        lr_step += 1

        if (step + 1) % 100 == 0:
            print(datetime.datetime.now())
            print(f'Epoch: {epoch + 1}')
            print(f'Step: {step + 1}')
            print(f'Number of samples seen: {(step + 1) * config.args.batch_size}')
            print(f"Loss is: {loss}\n")
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

    if (val_acc > best_acc or epoch == 0):
        best_acc = val_acc

    print(f"End of epoch {epoch + 1}")
    print(f"Validation accuracy: {float(val_acc)}")

    # Reset metrics
    train_loss.reset_states()
    valid_loss.reset_states()
    train_acc.reset_states()
    validation_acc.reset_states()

# End of training
print(f"Best accuracy is: {best_acc}")
model.summary()
