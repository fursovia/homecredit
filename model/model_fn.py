"""Определяем модель с помощью тф-ского estimator-а"""

import tensorflow as tf
from model.triplet_loss import batch_all_triplet_loss
from model.architectures import build_model


def model_fn(features, labels, mode, params):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        features: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        labels: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        # Compute the embeddings with the model
        embeddings = build_model(is_training, features, params)

    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)
 
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          export_outputs={
                                              'predict': tf.estimator.export.PredictOutput(predictions)
                                          })

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    loss, fraction, p_at_k = batch_all_triplet_loss(labels, embeddings, params)

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.variable_scope("metrics"):
            eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm),
                               "precision_at_K": tf.metrics.mean(p_at_k),
                               "fraction_positive_triplets": tf.metrics.mean(fraction)}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('fraction_positive_triplets', fraction)
    tf.summary.scalar('precision_at_K', p_at_k)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)

    global_step = tf.train.get_global_step()

    # train_op = optimizer.minimize(loss, global_step=global_step)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=params.learning_rate,
        optimizer=optimizer
    )

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
