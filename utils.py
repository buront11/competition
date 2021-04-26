import torch

import seaborn as sns

import pandas as pd

import os

def get_flights_dataset():
    flight_data = sns.load_dataset("flights")
    all_data = flight_data['passengers'].values.astype(float)

    test_data_size = 12
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

    train_window = 12
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    return train_inout_seq, test_data

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

def concat_csv(csv_paths,add_type=False):
    """複数のcsvファイルを結合する関数

    Parameters
    ----------
    csv_paths : list
        結合したいcsvファイルのパスのリスト

    Returns
    -------
    pandas.df
        csvファイルを結合してtypeを追加したpandasのデータフレーム
    """
    if add_type:
        df_concat = pd.read_csv(csv_paths[0])
        df_concat['type'] = os.path.splitext(os.path.basename(csv_paths[0]))[0]

        for path in csv_paths[1:]:
            df_add = pd.read_csv(path)
            df_add['type'] = os.path.splitext(os.path.basename(path))[0]
            df_concat = pd.concat([df_concat,df_add])
    else:
        df_concat = pd.read_csv(csv_paths[0])

        for path in csv_paths[1:]:
            df_add = pd.read_csv(path)
            df_concat = pd.concat([df_concat,df_add])

    return df_concat