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
from config import base_config as config
from architect import Architect
import datetime
from datasets import CIFAR10Input


LOG_DIR='./logs'

tf.get_logger().setLevel('INFO')
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

policy = tf.keras.mixed_precision.Policy("mixed_float16")

tf.keras.mixed_precision.set_global_policy(policy)    
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

class TrainableNetwork(Network):
    
    def __ini__(self, kwargs):
        super().__init__(**kwargs)

    def train_step(self, data):
        features, labels = data
        images, labels = features["image"], labels["label"]
        with tf.GradientTape() as tape:
            logits = self(images, training=True)
            logits = tf.cast(logits, tf.float32)
            loss = self.compiled_loss(labels, logits)
        self.optimizer.minimize(loss, self.trainable_weights, tape=tape)
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        features, labels = data
        images, labels = features["image"], labels["label"]
        logits = self(images, training=False)
        logits = tf.cast(logits, tf.float32)
        
        self.compiled_loss(labels, logits)
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}
    
class TrainableArchitect(keras.Model):
    def __init__(self, model, unrolled=False):
        super().__init__()
        self.model = model
    
    def train_step(self, data):
        self.model.train_step(data)
        features, labels = data
        images, labels = features["image"], labels["label"]
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(self.model.alphas_normal)
            tape.watch(self.model.alphas_reduce)
            logits = self.model(images, training=True)
            loss = self.model.compiled_loss(labels, logits)
        self.optimizer.minimize(loss, [self.model.alphas_normal, self.model.alphas_reduce], tape=tape)
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}
    def test_step(self, data):
        return self.model.test_step(data)


class LogAlphasCallback(keras.callbacks.Callback):
    def __init__(self,logdir):
        super().__init__()
        self.logdir = logdir
        self.alphas_writer = tf.summary.create_file_writer(self.logdir)
        
    
    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        with self.alphas_writer.as_default():
            tf.summary.histogram(f"alphas_vs_epoch", tf.nn.softmax(self.model.model.arch_params(), axis=-1), step=epoch)

class ExtendedTensorBoardCallback(keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_val_acc = 0.0
        self.best_genotype = None
        
    def on_epoch_end(self, epoch, logs=None):
        with open(f"{self.log_dir}/genotype_epoch", 'a') as genotype_file:
            genotype_file.write(f"{epoch+1}, " + str(self.model.model.genotypes()) + "\n")
        if logs["val_acc_top1"] >= self.best_val_acc:
            self.best_val_acc = logs["val_acc_top1"]
            self.best_genotype = self.model.model.genotypes()
        with open(f"{self.log_dir}/best_genotype", 'w') as genotype_file:
            genotype_file.write(str(self.model.model.genotypes()))




criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = TrainableNetwork(16, criterion, 10, 8, n_nodes=4, multiplier=4,
                auxiliary_skip=True, auxiliary_op='skip_connect')
architect = TrainableArchitect(model, False)

best_genotype = model.genotypes()

# prepare log directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"logs/{current_time}/"
train_log_dir = f"logs/{current_time}/search_arch" + '/train'
test_log_dir = f"logs/{current_time}/search_arch" + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

log_alphas_cb = LogAlphasCallback(logdir)
tensorboard_cb = ExtendedTensorBoardCallback(log_dir=logdir)


with open(f"{train_log_dir}/genotype_initial", 'w') as genotype_file:
    genotype_file.write(str(best_genotype))
with open(f"{train_log_dir}/config", 'w') as config_file:
    config_file.write(str(config))
print(f"Initial genotype: {best_genotype}")
print(f"Initial alphas: {tf.nn.softmax(model.arch_params(), axis=-1)}")

train = CIFAR10Input(True, image_size=config["train_image_size"], image_dtype=tf.float16, cache=False)
eval = CIFAR10Input(False, image_size=config["eval_image_size"], image_dtype=tf.float16, cache=False)

train_batch_size = config["train_batch_size"]
eval_batch_size = config["eval_batch_size"]

train_steps_per_epoch = 50000//train_batch_size
eval_steps_per_epoch = 10000//eval_batch_size

model.compile(optimizer=keras.optimizers.SGD(0.025),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=config["label_smoothing"]),
                metrics=[
                    tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="acc_top1")
                ]
                )

architect.compile(optimizer=keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=config["label_smoothing"]),
                metrics=[
                    tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="acc_top1")
                ]
                )

architect.fit(
        train.input_fn(params={"batch_size": train_batch_size}),
        epochs=5,
        steps_per_epoch=train_steps_per_epoch,
        validation_data=eval.input_fn(params={"batch_size": eval_batch_size}),
        validation_steps=eval_steps_per_epoch,
        verbose=1,
        callbacks=[log_alphas_cb, tensorboard_cb]
    )

# for epoch in range(config.args.epochs):
#     #model._aux_decay = architect.v_model._aux_decay = linear_decay(epoch)
#     # Training
#     for step, ((x_batch_train, y_batch_train), (x_batch_valid, y_batch_valid)) in enumerate(zip(train_dataset, val_dataset)):
#         # First build the model
#         #if epoch == 0 and step == 0:
#             #architect.v_model._loss(x_batch_valid, y_batch_valid)

#         lr = tf.cast(current_lr(lr_step, decay_steps, config.args.learning_rate_min, config.args.learning_rate), tf.float32)
#         architect_step(x_batch_train, y_batch_train, x_batch_valid, y_batch_valid)
#         loss = train_step(x_batch_train, y_batch_train)
#         lr_step += 1

#         if (step + 1) % 100 == 0:
#             print(datetime.datetime.now())
#             print(f'Epoch: {epoch + 1}')
#             print(f'Step: {step + 1}')
#             print(f'Number of samples seen: {(step + 1) * config.args.batch_size}')
#             print(f"Loss is: {loss}")
#             print(f"Learning rate: {lr}\n")


#     with train_summary_writer.as_default():
#         tf.summary.scalar('loss', train_loss.result(), step=epoch)
#         tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

#     trn_acc = train_acc.result()

#     # Validation
#     for step, (x_batch_valid, y_batch_valid) in enumerate(val_dataset):
#         loss = validation_step(x_batch_valid, y_batch_valid)
#         if (step + 1) % 100 == 0:
#             print(datetime.datetime.now())
#             print(f'Epoch: {epoch + 1}')
#             print(f'Validation step: {step + 1}')
#             print(f"Validation loss is: {loss}\n")

#     val_acc = validation_acc.result()

#     with test_summary_writer.as_default():
#         tf.summary.scalar('loss', valid_loss.result(), step=epoch)
#         tf.summary.scalar('accuracy', validation_acc.result(), step=epoch)

#     # End of epoch logging and updating/reseting metrics
#     with open(f"{train_log_dir}/genotype_epoch_{epoch + 1}", 'w') as genotype_file:
#         genotype_file.write(str(model.genotypes()))

#     if (val_acc > best_acc or epoch == 0):
#         best_acc = val_acc
#         best_genotype = model.genotypes()

#     print(f"End of epoch {epoch + 1}")
#     print(f"Validation accuracy: {float(val_acc)}")
#     print(f"Genotype: {model.genotypes()}")
#     print(f"Alphas: {tf.nn.softmax(model.arch_params(), axis=-1)}\n\n")

#     train_loss.reset_states()
#     valid_loss.reset_states()
#     train_acc.reset_states()
#     validation_acc.reset_states()

# # End of architecture search
# print(f"Best accuracy is: {best_acc}")
# print(f"This was achieved with this genotype: {best_genotype}")
# print(f"Alphas: {tf.nn.softmax(model.arch_params(), axis=-1)}")
model.summary()
with open(f"{train_log_dir}/genotype_best", 'w') as genotype_file:
    genotype_file.write(str(best_genotype))