"""Подготовка датасета"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/i.fursov/hc/data',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    args = parser.parse_args()

    print('Reading the data...')
    train = pd.read_csv('/data/i.fursov/hc/inputs/application_train.csv.zip')
    test = pd.read_csv('/data/i.fursov/hc/inputs/application_test.csv.zip')
    prev = pd.read_csv('/data/i.fursov/hc/inputs/previous_application.csv.zip')
    buro = pd.read_csv('/data/i.fursov/hc/inputs/bureau.csv.zip')
    buro_balance = pd.read_csv('/data/i.fursov/hc/inputs/bureau_balance.csv.zip')
    credit_card = pd.read_csv('/data/i.fursov/hc/inputs/credit_card_balance.csv.zip')
    POS_CASH = pd.read_csv('/data/i.fursov/hc/inputs/POS_CASH_balance.csv.zip')
    payments = pd.read_csv('/data/i.fursov/hc/inputs/installments_payments.csv.zip')
    submission = pd.read_csv('/data/i.fursov/hc/inputs/sample_submission.csv.zip')

    Y = train['TARGET']
    Y = Y.values
    del train['TARGET']

