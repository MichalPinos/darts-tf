from datasets import CIFAR10Input
import matplotlib.pyplot as plt
import numpy as np
from model_train import *
import tensorflow as tf
import re
from absl import app
from absl import flags
from config import base_config as config
import math

FLAGS=flags.FLAGS
flags.DEFINE_string('logdir', 'logdir/', 'Logging directory')
flags.DEFINE_string('load_weights', None, "Path to model weights.")


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class TrainableNetwork(Network):
    
    def __ini__(self, kwargs):
        super().__init__(**kwargs)
        
    def _reg_l2_loss(self, weight_decay=1e-5, regex=r'.*(kernel|weight):0$'):
        var_match = re.compile(regex)
        return weight_decay * tf.add_n([
            tf.nn.l2_loss(v)
            for v in self.trainable_variables
            if var_match.match(v.name)
    ])
    
    def train_step(self, data):
        features, labels = data
        images, labels = features["image"], labels["label"]
        with tf.GradientTape() as tape:
            logits, logits_aux = self(images, training=True)
            logits = tf.cast(logits, tf.float32)
            
            loss = self.compiled_loss(labels, logits,
                                      regularization_losses=[self._reg_l2_loss()])
            if self._auxiliary:
                logits_aux  = tf.cast(logits_aux, tf.float32)
                loss_aux = self.compiled_loss(labels, logits_aux)
                loss += loss_aux
                
        self.optimizer.minimize(loss, self.trainable_weights, tape=tape)
        self.compiled_metrics.update_state(labels, logits)
        self.compiled_metrics.update_state(labels, logits_aux)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        features, labels = data
        images, labels = features["image"], labels["label"]
        
        logits, _ = self(images, training=False)
        logits = tf.cast(logits, tf.float32)
        
        self.compiled_loss(labels, logits,
                           regularization_losses=[self._reg_l2_loss()])
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}
    

with open("/home/michalpinos/darts_tf2/darts-tf2/genotypes/og", "r") as f:
    genotype = f.read()
genotype = eval(genotype)
        

def main(argv):
        
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)    
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    model = TrainableNetwork(
        config["init_channels"], None, config["num_classes"], config["num_layers"], 
        n_nodes=4, multiplier=4, 
        genotype=genotype, drop_rate=config["dropout_rate"], 
        auxiliary=config["auxiliary"]
        )

    tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=FLAGS.logdir)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=FLAGS.logdir + "/weights.h5",
        save_weights_only=True,
        monitor='val_acc_top1',
        mode='max',
        save_best_only=True)
    
    
    train_batch_size = config["train_batch_size"]
    eval_batch_size = config["eval_batch_size"]
    
    # scale lr (initial_lr is set for batch_size 100)
    #config["lr_init"] = config["lr_init"] * (train_batch_size/100.0)
    
    train_steps_per_epoch = config["train_dataset_size"]//train_batch_size
    eval_steps_per_epoch = config["test_dataset_size"]//eval_batch_size
    
    decay_steps = train_steps_per_epoch * config["lr_decay_epoch"]
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(config["lr_init"], 
                                                                  decay_steps, 
                                                                  config["lr_decay_factor"], 
                                                                  staircase=True
                                                                  )
    # lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(config["lr_init"], 
    #                                                          config["epochs"] * train_steps_per_epoch, 
    #                                                          config["lr_min"], 
    #                                                               )

    if config["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(lr_scheduler)
    elif config["optimizer"] == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(lr_scheduler, rho=0.9, momentum=0.9, epsilon=1e-3)
    elif config["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(lr_scheduler)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported!")
    
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=config["label_smoothing"]),
                metrics=[
                    tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="acc_top1")
                ]
                )

    model.summary()
    model.build(input_shape=(None, config["train_image_size"], config["train_image_size"], 3))
    
    if FLAGS.load_weights:
        model.load_weights(FLAGS.load_weights, by_name=True, skip_mismatch=True)
        
    
    train = CIFAR10Input(True, image_size=config["train_image_size"], image_dtype=tf.float16, cache=False)
    eval = CIFAR10Input(False, image_size=config["eval_image_size"], image_dtype=tf.float16, cache=False)
    
    model.fit(
        train.input_fn(params={"batch_size": train_batch_size}),
        epochs=config["epochs"],
        steps_per_epoch=train_steps_per_epoch,
        validation_data=eval.input_fn(params={"batch_size": eval_batch_size}),
        validation_steps=eval_steps_per_epoch,
        verbose=1,
        callbacks=[tb_callback, model_checkpoint_callback]
    )
    
if __name__ == '__main__':
  app.run(main)