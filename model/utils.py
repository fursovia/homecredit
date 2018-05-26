"""Вспомогательные функции"""

import json
import tensorflow as tf
import numpy as np
import pickle
import os


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        self.num_epochs = None
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
        
        
def calculate_cosine_sim(inputs, precalculated_embeddings):
    """Compute the 2D matrix of cosine distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
    Returns:
        _pairwise_similarities: tensor of shape (batch_size, batch_size)
    """
    normalized_inputs = tf.nn.l2_normalize(inputs, axis=1)
    normalized_embeds = tf.nn.l2_normalize(precalculated_embeddings, axis=1)

    similarities = tf.matmul(normalized_inputs, normalized_embeds, adjoint_b=True)

    return similarities

