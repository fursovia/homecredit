"""Здесь перечислены все реализованные архитектуры. Выбрать архитектуру можно в файле params.json"""

import tensorflow as tf


def build_model(is_training, sentences, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """

    if params.architecture == 'dense':
        with tf.variable_scope('fc_1'):
            out = tf.layers.dense(sentences, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            out = tf.layers.dense(out, 128, activation=tf.nn.elu)

        if params.loss_type == 'ATnT':
            with tf.variable_scope('fc_3'):
                out = tf.layers.dense(out, 64, activation=tf.nn.sigmoid)
                # out = tf.layers.dense(out, 128)
                # out = tf.nn.l2_normalize(out, axis=1)
        else:
            with tf.variable_scope('fc_3'):
                out = tf.layers.dense(out, 32)
        return out

    if params.architecture == 'dense_batchnorm':
        with tf.variable_scope('fc_1'):
            out = tf.layers.dense(sentences, 256)
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.elu(out)

        with tf.variable_scope('fc_2'):
            out = tf.layers.dense(out, 128)
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.elu(out)

        if params.loss_type == 'ATnT':
            with tf.variable_scope('fc_3'):
                out = tf.layers.dense(out, 64, activation=tf.nn.sigmoid)
                # out = tf.layers.dense(out, 128)
                # out = tf.nn.l2_normalize(out, axis=1)
        else:
            with tf.variable_scope('fc_3'):
                out = tf.layers.dense(out, 64)
        return out