import os
import sys
import json

import pandas as pd
import joblib
import torch

sys.path.append('../../')

import preprocess_base
from preprocess_base import PreprocessBase
import utils

def preprocess():
    train_data = preprocess_base.load_tsv('../../data/advitisement/train.tsv')
    test_data = preprocess_base.load_tsv('../../data/advitisement/test.tsv')

    # 欠損値のチェック
    pb = PreprocessBase(train_data, test_data)

    pb.train_data = pb.train_data.fillna(-1)
    pb.train_data = category2count(pb.train_data)
    pb.train_data = wh2area(pb.train_data)

    pb.test_data = pb.test_data.fillna(-1)
    pb.test_data = category2count(pb.test_data)
    pb.test_data = wh2area(pb.test_data)
    pb.standard_scaler(['height','width','aratio','local','area'])

    train_input = torch.tensor(pb.train_data.loc[:,['height','width','aratio','local','area', 'category']].values).float()
    input_dim = train_input.size(1)
    train_label = torch.tensor(pb.train_data['Y'].values.tolist()).long()

    test_index = torch.tensor(pb.test_data['id'].values.tolist())
    test_input = torch.tensor(pb.test_data.loc[:,['height','width','aratio','local','area', 'category']].values).float()

    train_dataset = preprocess_base.tensor_dataset(train_input, train_label)
    test_dataset = [[i,j] for i , j in zip(test_index, test_input)]

    joblib.dump(train_dataset, '../../data/advitisement/train')
    joblib.dump([train_input, train_label],'../../data/advitisement/train_split')
    joblib.dump(test_dataset,'../../data/advitisement/test')

    utils.update_json('./hyper_parameters.json', {'input_dim':input_dim})

def category2count(df):
    column = list(range(5,1559))
    df_category = df.iloc[:,column]

    catecory_info = []

    for index, row in df_category.iterrows():
        value = sum(row)
        if value:
            catecory_info.append(value)
        else:
            catecory_info.append(0)

    df['category'] = catecory_info

    return df

def wh2area(df):
    df['area'] = df['width'].copy()*df['height'].copy()

    return df

if __name__=='__main__':
    preprocess()