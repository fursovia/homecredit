"""Инпуты для модели с помощью tf dataset api"""


import pickle
import tensorflow as tf
import os
import numpy as np


def data_generator(data, labels, num_classes, num_examples):
    """Функция генерит батчи
    Args:
        data:
        labels:
    Returns:
        batch:
    """

    unique_labels = np.unique(labels)

    for _ in range(1000):
        selected_labels = np.random.choice(unique_labels, num_classes, replace=False)
        data_ = []

        for y in selected_labels:
            selected_data = np.random.choice(data[selected_labels == y], num_examples, replace=True)
            data_.extend(list(selected_data))
        yield data_


def train_input_fn(data_dir, params):
    """
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    data_ = pickle.load(open(os.path.join(data_dir, 'train/X_tr.pkl'), 'rb'))
    labels_ = pickle.load(open(os.path.join(data_dir, 'train/Y_tr.pkl'), 'rb'))

    labels_ = np.array(labels_, int)

    params.train_size = len(data_)

    dataset1 = tf.data.Dataset.from_tensor_slices(data_)
    dataset2 = tf.data.Dataset.from_tensor_slices(labels_)
    dataset = tf.data.Dataset.zip((dataset1, dataset2))

    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(buffer_size=None) # auto detection of optimal buffer size.
    return dataset

def test_input_fn(data_dir, params):
    """
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    data_ = pickle.load(open(os.path.join(data_dir, 'eval/X_ev.pkl'), 'rb'))
    labels_ = pickle.load(open(os.path.join(data_dir, 'eval/Y_ev.pkl'), 'rb'))
    labels_ = np.array(labels_, int)
    params.train_size = len(data_)

    dataset1 = tf.data.Dataset.from_tensor_slices(data_)
    dataset2 = tf.data.Dataset.from_tensor_slices(labels_)
    dataset = tf.data.Dataset.zip((dataset1, dataset2))

    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(buffer_size=None) # auto detection of optimal buffer size.

    return dataset

def final_train_input_fn(data_dir, params):
    """
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    data_ = pickle.load(open(os.path.join(data_dir, 'X.pkl'), 'rb'))
    labels_ = pickle.load(open(os.path.join(data_dir, 'Y.pkl'), 'rb'))

    params.train_size = len(data_)

    dataset1 = tf.data.Dataset.from_tensor_slices(data_)
    dataset2 = tf.data.Dataset.from_tensor_slices(labels_)
    dataset = tf.data.Dataset.zip((dataset1, dataset2))

    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(buffer_size=None) # auto detection of optimal buffer size.
    return dataset

def balanced_train_input_fn(data_dir, params):
    """
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    data_ = pickle.load(open(os.path.join(data_dir, 'train/X_tr.pkl'), 'rb'))
    labels_ = pickle.load(open(os.path.join(data_dir, 'train/Y_tr.pkl'), 'rb'))

    dataset = tf.data.Dataset.from_generator(data_generator(data_, labels_, 3, 4),
                                             tf.int32)

    value = dataset.make_one_shot_iterator().get_next()

    return value
