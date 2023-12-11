"""
Module providing Architect class which is used for architecture optimization

This code is part of reimplementation of original DARTS
and is based on it, this part is inspired by newer implementation,
which can be found here: https://github.com/khanrc/pt.darts/
and is licensed under MIT license

Author: Vojtech Eichler
Date: April 2023
"""

from model_search import Network
import tensorflow as tf
import tensorflow.keras as keras

class Architect():
    """Architect class for architecture optimization
    """
    def __init__(self, model, args, criterion):
        """Architect initialization function

        Args:
            model : Model whom architecture is being optimized
            args : Object storing current config
            criterion : Loss function
        """
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.model = model
        #self.v_model = Network(args.init_channels, criterion, 10, args.layers, n_nodes=args.nodes, multiplier=args.multiplier, auxiliary_skip=True, auxiliary_op='skip_connect')
        #self.v_model.set_weights(self.model.get_weights())
        self.optimizer = keras.optimizers.Adam(learning_rate=args.arch_learning_rate, beta_1=0.5, beta_2=0.999)#, weight_decay=1e-3)

    def step(self, x_train, y_train, x_valid, y_valid, xi, net_optimizer, unrolled):
        """Method performing one architecture optimization step

        Args:
            x_train : Train images
            y_train : Train annotations
            x_valid : Validation images
            y_valid : Validation annotations
            xi : Current learning rate
            net_optimizer : Optimizer for regular layer weights
            unrolled : Specify whether to use unrolled step
        """
        if unrolled:
            # Perform unrolled step by approximating weights with one training step
            grads_normal, grads_reduce = self._backward_step_unrolled(x_train, y_train, x_valid, y_valid, xi, net_optimizer)
        else:
            # Don't approximate weights with training step and just calculate gradients
            with tf.GradientTape(persistent=True) as gt:
                gt.watch(self.model.alphas_normal)
                gt.watch(self.model.alphas_reduce)
                loss = self._backward_step(x_valid, y_valid)
            grads_normal = gt.gradient(loss, self.model.alphas_normal)
            grads_reduce = gt.gradient(loss, self.model.alphas_reduce)
        # Apply weight decay
        # self.model.alphas_normal.assign_sub(self.model.alphas_normal * 1e-3 * xi)
        # self.model.alphas_reduce.assign_sub(self.model.alphas_reduce * 1e-3 * xi)
        self.optimizer.apply_gradients(zip([grads_normal, grads_reduce], [self.model.alphas_normal, self.model.alphas_reduce]))

    def _backward_step_unrolled(self, x_train, y_train, x_valid, y_valid, xi, net_optimizer):
        """Method which performs one step unrolled optimization step

        Args:
            x_train : Train images
            y_train : Train annotations
            x_valid : Validation images
            y_valid : Validation annotations
            xi : Current learning rate
            net_optimizer : Optimizer for regular layer weights

        Returns:
            Gradients for normal and reduction cell architectures
        """
        self._virtual_step(x_train, y_train, xi, net_optimizer)

        with tf.GradientTape() as gt:
            gt.watch(self.v_model.alphas_normal)
            gt.watch(self.v_model.alphas_reduce)
            loss = self.v_model._loss(x_valid, y_valid)

        variables = self.v_model.trainable_weights
        variables.append(self.v_model.alphas_normal)
        variables.append(self.v_model.alphas_reduce)
        v_grads = gt.gradient(loss, variables)

        dalpha = v_grads[-2:] # Architecture weights gradients
        dw = v_grads[:-2] # Layers weights gradients

        hess = self.calc_hessian(dw, x_train, y_train)
        hess_normal, hess_reduce = tf.split(hess, num_or_size_splits=2, axis=0)

        # Compute gradients
        grads_normal = tf.math.subtract(dalpha[0], tf.multiply(xi, hess_normal))
        grads_reduce = tf.math.subtract(dalpha[1], tf.multiply(xi, hess_reduce))

        return [grads_normal, grads_reduce]

    def calc_hessian(self, dw, x_train, y_train):
        """Calculate hessian for architecture weights

        Args:
            dw : Gradients for weights of layers
            x_train : Train images
            y_train : Train annotations

        Returns:
            Hessian
        """
        norm = tf.concat([tf.reshape(x, [-1]) for x in dw], 0)
        norm = tf.norm(norm)
        eps = tf.math.divide(0.01, norm)

        # Positive gradients
        for idx, d in enumerate(dw):
            self.model.trainable_weights[idx].assign_add(tf.math.multiply(eps, d))

        with tf.GradientTape(persistent=True) as gt:
            gt.watch(self.model.alphas_normal)
            gt.watch(self.model.alphas_reduce)
            loss = self.model._loss(x_train, y_train, training=True)
        dalpha_positive_norm = gt.gradient(loss, self.model.alphas_normal)
        dalpha_positive_red = gt.gradient(loss, self.model.alphas_reduce)

        # Negative gradients
        for idx, d in enumerate(dw):
            self.model.trainable_weights[idx].assign_add(tf.math.multiply(tf.math.multiply(-2., eps), d))

        with tf.GradientTape(persistent=True) as gt:
            gt.watch(self.model.alphas_normal)
            gt.watch(self.model.alphas_reduce)
            loss = self.model._loss(x_train, y_train, training=True)
        dalpha_negative_norm = gt.gradient(loss, self.model.alphas_normal)
        dalpha_negative_red = gt.gradient(loss, self.model.alphas_reduce)

        dalpha_positive = tf.concat([dalpha_positive_norm, dalpha_positive_red], 0)
        dalpha_negative = tf.concat([dalpha_negative_norm, dalpha_negative_red], 0)

        # Restore weights
        for idx, d in enumerate(dw):
            self.model.trainable_weights[idx].assign_add(tf.math.multiply(eps, d))

        # Calculate hessian
        hess = tf.math.divide(tf.math.subtract(dalpha_positive, dalpha_negative), tf.math.multiply(2., eps))
        return hess

    def _virtual_step(self, x_train, y_train, xi, net_optimizer):
        """Method which manually performs one training step and updates virtual model weights

        Args:
            x_train : Train images
            y_train : Train annotations
            xi : Current learning rate
            net_optimizer : Optimizer for regular layer weights
        """
        with tf.GradientTape() as gt:
            loss = self.model._loss(x_train, y_train, training=True)
        grads = gt.gradient(loss, self.model.trainable_weights)

        # Get optimizer weights
        moment = net_optimizer.variables()[1:]
        if not moment: # In first step optimizer doesn't have weights initialized yet
            moment = [0] * len(grads)

        # Perform one training step
        for idx, (w, m, g) in enumerate(zip(self.model.trainable_weights, moment, grads)):
            self.v_model.trainable_weights[idx].assign(w - xi * ( m * self.momentum + g + self.weight_decay * w))

        # Synchornize architecture weights
        for idx, a in enumerate(self.model._arch_params):
            self.v_model._arch_params[idx].assign(a)

    def _backward_step(self, input_valid, target_valid):
        return self.model._loss(input_valid, target_valid, training=True)