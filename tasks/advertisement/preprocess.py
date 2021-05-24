import os
import sys
import json

import pandas as pd
import joblib
import torch

sys.path.append('../../')

import preprocess_base
import utils

def preprocess():
    train_data = preprocess_base.load_tsv('../../data/advitisement/train.tsv')
    test_data = preprocess_base.load_tsv('../../data/advitisement/test.tsv')

    train_data = preprocess_base.drop_nan(train_data)
    train_data = category2onehot(train_data)
    train_input = torch.tensor(train_data.loc[:,['height','width','aratio','local','category']].values).float()
    input_dim = train_input.size(1)
    train_label = torch.tensor(train_data['Y'].values.tolist()).long()

    test_data = category2onehot(test_data)
    test_index = torch.tensor(test_data['id'].values.tolist())
    test_input = torch.tensor(test_data.loc[:,['height','width','aratio','local','category']].values).float()

    train_dataset = preprocess_base.tensor_dataset(train_input, train_label)
    test_dataset = [[i,j] for i , j in zip(test_index, test_input)]

    joblib.dump(train_dataset,'../../data/advitisement/train')
    joblib.dump(test_dataset,'../../data/advitisement/test')

    utils.update_json('./hyper_parameters.json', {'input_dim':input_dim})

def category2onehot(df):
    column = list(range(5,1559))
    df_category = df.iloc[:,column]

    catecory_info = []

    for index, row in df_category.iterrows():
        value = sum(row)
        if value:
            catecory_info.append(1)
        else:
            catecory_info.append(0)

    df['category'] = catecory_info

    return df

if __name__=='__main__':
    preprocess()