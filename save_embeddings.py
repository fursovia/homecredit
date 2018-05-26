"""Позволяет перевести данные в эмбединги с помощью предобученной модели"""

import tensorflow as tf
import argparse
import os
from model.utils import Params
from model.model_fn import model_fn
from model.input_fn import final_train_input_fn
from model.input_fn import train_input_fn
from model.input_fn import eval_input_fn
from model.input_fn import test_input_fn
import pickle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/data/i.fursov/hc/experiments/',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='/data/i.fursov/hc/data',
                    help="Directory containing the dataset")
parser.add_argument('--test_time', default='N',
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

    if args.test_time == 'N':
        ### ПОЛУЧАЕМ ВСЕ ЭМБЕДИНГИ ИЗ ТРЕЙНА
        tf.logging.info("Predicting train/test data...")
        train_path = os.path.join(args.data_dir, 'train/X_tr.pkl')
        eval_path = os.path.join(args.data_dir, 'eval/X_ev.pkl')
        assert os.path.isfile(train_path), "No file found at {}".format(train_path)
        assert os.path.isfile(eval_path), "No file found at {}".format(eval_path)

        train_data = pickle.load(open(train_path, 'rb'))
        eval_data = pickle.load(open(eval_path, 'rb'))

        # train embeddings
        train_predictions = estimator.predict(lambda: train_input_fn(args.data_dir, params))

        train_embeddings = np.zeros((len(train_data), 64))
        for i, p in enumerate(train_predictions):
            train_embeddings[i] = p['embeddings']

        # eval embeddings
        eval_predictions = estimator.predict(lambda: eval_input_fn(args.data_dir, params))

        eval_embeddings = np.zeros((len(eval_data), 64))
        for i, p in enumerate(eval_predictions):
            eval_embeddings[i] = p['embeddings']

        tf.logging.info("Saving embeddings...")
        train_emb_path = os.path.join(args.model_dir, 'train_embeddings.pkl')
        eval_emb_path = os.path.join(args.model_dir, 'eval_embeddings.pkl')

        pickle.dump(train_embeddings, open(train_emb_path, 'wb'))
        pickle.dump(eval_embeddings, open(eval_emb_path, 'wb'))
    else:
        ### ПОЛУЧАЕМ ВСЕ ЭМБЕДИНГИ ИЗ ТРЕЙНА
        tf.logging.info("Predicting train data...")

        train_path = os.path.join(args.data_dir, 'X.pkl')
        test_path = os.path.join(args.data_dir, 'test/X_te.pkl')
        assert os.path.isfile(train_path), "No file found at {}".format(train_path)
        assert os.path.isfile(test_path), "No file found at {}".format(test_path)

        train_data = pickle.load(open(train_path, 'rb'))
        test_data = pickle.load(open(test_path, 'rb'))

        # train embeddings
        train_predictions = estimator.predict(lambda: final_train_input_fn(args.data_dir, params))

        train_embeddings = np.zeros((len(train_data), 64))
        for i, p in enumerate(train_predictions):
            train_embeddings[i] = p['embeddings']

        # test embeddings
        test_predictions = estimator.predict(lambda: test_input_fn(args.data_dir, params))

        test_embeddings = np.zeros((len(test_data), 64))
        for i, p in enumerate(test_predictions):
            test_embeddings[i] = p['embeddings']

        tf.logging.info("Saving embeddings...")
        train_emb_path = os.path.join(args.model_dir, 'train_embeddings.pkl')
        test_emb_path = os.path.join(args.model_dir, 'test_embeddings.pkl')

        pickle.dump(train_embeddings, open(train_emb_path, 'wb'))
        pickle.dump(test_embeddings, open(test_emb_path, 'wb'))
