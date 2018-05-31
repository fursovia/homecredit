"""Подготовка датасета"""

import pandas as pd
import numpy as np
import pickle
import argparse
import gc
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/i.fursov/hc',
                    help="Directory containing the dataset")
parser.add_argument('--sample', default='N',
                    help="Directory containing the dataset")


def feature_engineering(app_both,
                        previous_application,
                        credit_card_balance,
                        bureau,
                        POS_CASH_balance,
                        installments_payments):
    """Генерим фичи, объединяем таблицы
    Args:
        ...
    Returns:
        merged_df: таблица со всеми фичами
    """
    # предыдущие кредиты
    print('Shape before merging with previous apps num data = {}'.format(app_both.shape))
    agg_funs = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum'}  # num of credits and total_amount
    prev_apps = previous_application.groupby('SK_ID_CURR').agg(agg_funs)
    prev_apps.columns = ['PREV_APP_COUNT', 'TOTAL_PREV_LOAN_AMT']
    merged_df = app_both.merge(prev_apps, left_on='SK_ID_CURR', right_index=True, how='left')
    print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))

    prev_apps_avg = previous_application.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(prev_apps_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_PAVG'])
    print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))

    prev_app_df, cat_feats, _ = process_dataframe(previous_application)
    prev_apps_cat_avg = prev_app_df[cat_feats + ['SK_ID_CURR']].groupby('SK_ID_CURR')\
        .agg({k: lambda x: str(x.mode().iloc[0]) for k in cat_feats})
    merged_df = merged_df.merge(prev_apps_cat_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_BAVG'])
    print('Shape after merging with previous apps cat data = {}'.format(merged_df.shape))

    wm = lambda x: np.average(x, weights=-1 / credit_card_balance.loc[x.index, 'MONTHS_BALANCE'])
    credit_card_avgs = credit_card_balance.groupby('SK_ID_CURR').agg(wm)
    merged_df = merged_df.merge(credit_card_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CCAVG'])

    most_recent_index = credit_card_balance.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = credit_card_balance.columns[credit_card_balance.dtypes == 'object'].tolist() + ['SK_ID_CURR']
    merged_df = merged_df.merge(credit_card_balance.loc[most_recent_index, cat_feats],
                                left_on='SK_ID_CURR',
                                right_on='SK_ID_CURR',
                                how='left',
                                suffixes=['', '_CCAVG'])
    print('Shape after merging with credit card data = {}'.format(merged_df.shape))

    credit_bureau_avgs = bureau.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(credit_bureau_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_BAVG'])
    print('Shape after merging with credit bureau data = {}'.format(merged_df.shape))

    wm = lambda x: np.average(x, weights=-1/POS_CASH_balance.loc[x.index, 'MONTHS_BALANCE'])
    f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF': wm}
    cash_avg = POS_CASH_balance.groupby('SK_ID_CURR')\
        ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 'SK_DPD', 'SK_DPD_DEF'].agg(f)
    merged_df = merged_df.merge(cash_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CAVG'])

    most_recent_index = POS_CASH_balance.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = POS_CASH_balance.columns[POS_CASH_balance.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    merged_df = merged_df.merge(POS_CASH_balance.loc[most_recent_index, cat_feats],
                                left_on='SK_ID_CURR',
                                right_on='SK_ID_CURR',
                                how='left',
                                suffixes=['', '_CAVG'])
    print('Shape after merging with pos cash data = {}'.format(merged_df.shape))

    ins_avg = installments_payments.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(ins_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_IAVG'])
    print('Shape after merging with installments data = {}'.format(merged_df.shape))

    return merged_df


def process_dataframe(input_df, encoder_dict=None):
    """Deal with categorical features"""

    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    categorical_feats = categorical_feats
    encoder_dict = {}
    for feat in categorical_feats:
        encoder = LabelEncoder()
        # TODO: нужно ли nan-ы заменять на новый класс?
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
        encoder_dict[feat] = encoder

    return input_df, categorical_feats.tolist(), encoder_dict


if __name__ == '__main__':
    args = parser.parse_args()

    # ПОДГРУЖАЕМ ДАННЫЕ
    print('Reading the data...')
    if args.sample == 'Y':
        sample_size = 1000
        save_path = os.path.join(args.data_dir, 'sample')
    else:
        sample_size = None
        save_path = os.path.join(args.data_dir, 'data')

    application_train = pd.read_csv(os.path.join(args.data_dir, 'inputs/application_train.csv.zip'), nrows=sample_size)
    application_test = pd.read_csv(os.path.join(args.data_dir, 'inputs/application_test.csv.zip'), nrows=sample_size)
    previous_application = pd.read_csv(os.path.join(args.data_dir, 'inputs/previous_application.csv.zip'), nrows=sample_size)
    bureau = pd.read_csv(os.path.join(args.data_dir, 'inputs/bureau.csv.zip'), nrows=sample_size)
    bureau_balance = pd.read_csv(os.path.join(args.data_dir, 'inputs/bureau_balance.csv.zip'), nrows=sample_size)
    credit_card_balance = pd.read_csv(os.path.join(args.data_dir, 'inputs/credit_card_balance.csv.zip'), nrows=sample_size)
    POS_CASH_balance = pd.read_csv(os.path.join(args.data_dir, 'inputs/POS_CASH_balance.csv.zip'), nrows=sample_size)
    installments_payments = pd.read_csv(os.path.join(args.data_dir, 'inputs/installments_payments.csv.zip'), nrows=sample_size)
    sample_submission = pd.read_csv(os.path.join(args.data_dir, 'inputs/sample_submission.csv.zip'), nrows=sample_size)

    # ГЕНЕРИМ ФИЧИ
    len_train = len(application_train)
    app_both = pd.concat([application_train, application_test])
    merged_df = feature_engineering(app_both,
                                    previous_application,
                                    credit_card_balance,
                                    bureau,
                                    POS_CASH_balance,
                                    installments_payments)

    meta_cols = ['SK_ID_CURR']
    meta_df = merged_df[meta_cols]
    merged_df.drop(columns=meta_cols, inplace=True)

    merged_df, categorical_feats, encoder_dict = process_dataframe(input_df=merged_df)

    labels = merged_df.pop('TARGET')
    labels = np.array(labels[:len_train], int)

    # УДАЛЯЕМ КАТЕГОРИАЛЬНЫЕ ФИЧИ
    dangerous_feats = []
    for feat in categorical_feats:
        feat_cardinality = len(merged_df[feat].unique())
        if (feat_cardinality > 10) & (feat_cardinality <= 100):
            print('Careful: {} has {} unique values'.format(feat, feat_cardinality))
        if feat_cardinality > 100:
            categorical_feats.remove(feat)
            dangerous_feats.append(feat)
            print('Dropping feat {} as it has {} unique values'.format(feat, feat_cardinality))
    merged_df.drop(columns=dangerous_feats, inplace=True)
    merged_df = pd.get_dummies(data=merged_df, columns=categorical_feats)
    print('Shape after one-hot encoding = {}'.format(merged_df.shape))

    # УДАЛЯЕМ ФИЧИ С NANs
    null_counts = merged_df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    null_ratios = null_counts / len(merged_df)

    null_thresh = 0.5
    null_cols = null_ratios[null_ratios > null_thresh].index
    merged_df.drop(columns=null_cols, inplace=True)
    print('Columns dropped for being over {}% null:'.format(100 * null_thresh))
    # for col in null_cols:
    #     print(col)

    # ЗАМЕНЯЕМ НАНЫ МЕДИАННЫМИ ЗНАЧЕНИЯМИ
    merged_df.fillna(merged_df.median(), inplace=True)

    # СКЕЙЛИМ ФИЧИ
    scaler = StandardScaler()
    merged_df = scaler.fit_transform(merged_df)

    X = merged_df[:len_train]
    X_te = merged_df[len_train:]

    del merged_df
    gc.collect()

    # ДЕЛИМ И СОХРАНЯЕМ
    X_tr, X_ev, Y_tr, Y_ev = train_test_split(X, labels,
                                              stratify=labels,
                                              test_size=0.1,
                                              random_state=43)

    pickle.dump(X_tr, open(os.path.join(save_path, 'train/X_tr.pkl'), 'wb'))
    pickle.dump(Y_tr, open(os.path.join(save_path, 'train/Y_tr.pkl'), 'wb'))
    pickle.dump(X_ev, open(os.path.join(save_path, 'eval/X_ev.pkl'), 'wb'))
    pickle.dump(Y_ev, open(os.path.join(save_path, 'eval/X_tr.pkl'), 'wb'))
    pickle.dump(X, open(os.path.join(save_path, 'X.pkl'), 'wb'))
    pickle.dump(labels, open(os.path.join(save_path, 'Y.pkl'), 'wb'))
    pickle.dump(X_te, open(os.path.join(save_path, 'test/X_te.pkl'), 'wb'))
