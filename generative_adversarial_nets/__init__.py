
from hbconfig import Config
import tensorflow as tf



class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self, inputs):

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            real_output = None

            sample_z = inputs
            fake_output = self._build_generator(sample_z)
        else:
            # Regularization Parameter
            real_output = self._build_discriminator(inputs)

            batch_size = tf.shape(inputs)[0]
            sample_z = tf.random_normal(
                [batch_size, Config.model.z_dim], mean=0, stddev=1, dtype=self.dtype, name="sample-z")

            fake = self._build_generator(sample_z)
            fake_output = self._build_discriminator(fake)

        return real_output, fake_output

    def _build_discriminator(self, inputs):
        with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(inputs, Config.model.encoder_h1,
                                 activation=tf.nn.elu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            h2 = tf.layers.dense(h1, Config.model.encoder_h2,
                                 activation=tf.nn.elu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            h3 = tf.layers.dense(h2, Config.model.encoder_h3,
                                 activation=tf.nn.elu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            output = tf.layers.dense(h3, 2)
        return output

    def _build_generator(self, sample_z):
        with tf.variable_scope('Generator'):
            h1 = tf.layers.dense(sample_z, Config.model.decoder_h1,
                                 activation=tf.nn.elu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            h2 = tf.layers.dense(h1, Config.model.decoder_h2,
                                 activation=tf.nn.elu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            h3 = tf.layers.dense(h2, Config.model.decoder_h3,
                                 activation=tf.nn.elu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            output = tf.layers.dense(h3, Config.model.n_output,
                                     activation=tf.nn.sigmoid)
        return output
