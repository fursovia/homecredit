"""Позволяет перевести данные в эмбединги с помощью предобученной модели"""

import tensorflow as tf
import argparse
import os
from model.utils import Params
from model.model_fn import model_fn
from model.utils import calculate_cosine_sim
import pickle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/data/i.fursov/experiments/',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/chars',
                    help="Directory containing the dataset")
parser.add_argument('--final_train', default='N',
                    help="Whether to train on a whole dataset")

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # ПОДГРУЖАЕМ ПАРАМЕТРЫ
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    ### ОПРЕДЕЛЯЕМ МОДЕЛЬ
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    if args.final_train == 'N':
        ### ПОЛУЧАЕМ ВСЕ ЭМБЕДИНГИ ИЗ ТРЕЙНА
        tf.logging.info("Predicting train/test data...")
        train_path = os.path.join(args.data_dir, 'train/X_tr.pkl')
        assert os.path.isfile(json_path), "No file found at {}".format(json_path)

        test_path = os.path.join(args.data_dir, 'test/X_te.pkl')
        assert os.path.isfile(json_path), "No file found at {}".format(json_path)

        train_data = pickle.load(open(train_path, 'rb'))
        test_data = pickle.load(open(test_path, 'rb'))

        # train embeddings
        train_input_fn = tf.estimator.inputs.numpy_input_fn(train_data,
                                                            num_epochs=1,
                                                            batch_size=512,
                                                            shuffle=False)
        train_predictions = estimator.predict(train_input_fn)

        train_embeddings = np.zeros((len(train_data), 128))
        for i, p in enumerate(train_predictions):
            train_embeddings[i] = p['embeddings']

        # test embeddings
        test_input_fn = tf.estimator.inputs.numpy_input_fn(test_data,
                                                           num_epochs=1,
                                                           batch_size=512,
                                                           shuffle=False)
        test_predictions = estimator.predict(test_input_fn)

        test_embeddings = np.zeros((len(test_data), 128))
        for i, p in enumerate(test_predictions):
            test_embeddings[i] = p['embeddings']

        tf.logging.info("Saving embeddings...")
        train_emb_path = os.path.join(args.model_dir, 'train_embeddings.pkl')
        test_emb_path = os.path.join(args.model_dir, 'test_embeddings.pkl')

        pickle.dump(train_embeddings, open(train_emb_path, 'wb'))
        pickle.dump(test_embeddings, open(test_emb_path, 'wb'))
    else:
        ### ПОЛУЧАЕМ ВСЕ ЭМБЕДИНГИ ИЗ ТРЕЙНА
        tf.logging.info("Predicting train/test data...")
        train_path = os.path.join(args.data_dir, 'X.pkl')
        assert os.path.isfile(json_path), "No file found at {}".format(json_path)
        train_data = pickle.load(open(train_path, 'rb'))

        train_input_fn = tf.estimator.inputs.numpy_input_fn(train_data,
                                                            num_epochs=1,
                                                            batch_size=512,
                                                            shuffle=False)
        train_predictions = estimator.predict(train_input_fn)

        train_embeddings = np.zeros((len(train_data), 128))
        for i, p in enumerate(train_predictions):
            train_embeddings[i] = p['embeddings']

        tf.logging.info("Saving embeddings...")
        train_emb_path = os.path.join(args.model_dir, 'train_embeddings.pkl')
        pickle.dump(train_embeddings, open(train_emb_path, 'wb'))