from __future__ import print_function


from hbconfig import Config
import tensorflow as tf

import generative_adversarial_nets



class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self.loss, self.train_op, self.metrics, self.predictions = None, None, None, None
        self._init_placeholder(features, labels)
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions={"prediction": self.predictions})

    def _init_placeholder(self, features, labels):
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.inputs = features["latent_vector"]
        else:
            self.inputs = features
            if type(features) == dict:
                self.inputs = features["input_data"]
            self.targets = labels

    def build_graph(self):
        graph = generative_adversarial_nets.Graph(self.mode)
        real_output, fake_output = graph.build(inputs=self.inputs)

        self._build_prediction(fake_output)
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(real_output, fake_output)
            self._build_optimizer()
            self._build_metric()

    def _build_prediction(self, output):
        self.predictions = output

    def _build_loss(self, real_output, fake_output):
        with tf.variable_scope('loss'):
            with tf.variable_scope('discriminator'):
                self.discriminator_loss = (tf.losses.sigmoid_cross_entropy(real_output, tf.ones(tf.shape(real_output))) +
                                    tf.losses.sigmoid_cross_entropy(fake_output, tf.zeros(tf.shape(fake_output))))
            with tf.variable_scope('generator'):
                self.generator_loss = tf.losses.sigmoid_cross_entropy(fake_output, tf.ones(tf.shape(fake_output)))

            self.total_loss = tf.add(self.discriminator_loss, self.generator_loss, name="total")

        tf.summary.scalar("loss/discriminator", self.discriminator_loss)
        tf.summary.scalar("loss/generator", self.generator_loss)
        tf.summary.scalar("loss/total", self.total_loss)

    def _build_optimizer(self):
        self.discriminator_train_op = tf.contrib.layers.optimize_loss(
            self.discriminator_loss, tf.train.get_global_step(),
            optimizer=Config.train.get('optimizer', 'Adam'),
            learning_rate=Config.train.learning_rate,
            summaries=['gradients', 'learning_rate'],
            name="discriminator_train_op")

        self.generator_train_op = tf.contrib.layers.optimize_loss(
            self.generator_loss, tf.train.get_global_step(),
            optimizer=Config.train.get('optimizer', 'Adam'),
            learning_rate=Config.train.learning_rate,
            summaries=['gradients', 'learning_rate'],
            name="generator_train_op")

        self.train_op = (self.discriminator_train_op, self.generator_train_op)

    def _build_metric(self):
        self.metrics = {}
