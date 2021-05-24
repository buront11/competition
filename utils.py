import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold

import numpy as np
import seaborn as sns
import pandas as pd

import os
import json
import subprocess

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """

        Parameters
        ----------
        patience : int, optional
            lossの更新が何回なければ学習を止めるか, by default 5
        verbose : bool, optional
            カウンターとスコアの表示の有無, by default False
        path : str, optional
            モデルの保存先, by default 'checkpoint_model.pth'
        """

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

def gen_hyper_param_template(save_dir, **kwargs):
    """hyper parameter.jsonのtemplateを作成するプログラム
    kwardsに任意の引数を与えることであらかじめ追加することが可能

    Parameters
    ----------
    save_dir : string
        保存先のディレクトリ
    """
    template_dict = {kwargs}

    with open(save_dir+"/hyper_paramter.json", 'w') as f:
        json.dump(template_dict,f)

def update_json(json_file, dict):
    """jsonファイルをupdateするプログラム
    
    Parameters
    ----------
    json_file : str
        jsonファイルのpath
    dict : dict
        追加もしくは更新したいdict
    """
    with open(json_file) as f:
        df = json.load(f)

    df.update(dict)

    with open(json_file, 'w') as f:
        json.dump(df, f, indent=4)

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

# gitのversionを吐くプログラム

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys='utilization.gpu', no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    gpu_info =  [{ k: v for k, v in zip(keys, line.split(', ')) } for line in lines]