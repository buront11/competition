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
    train_input = torch.tensor(train_data.loc[:,['id','height','width','aratio','local']].values).float()
    input_dim = train_input.size(1)
    train_label = torch.tensor(train_data['Y'].values.tolist()).long()

    test_input = preprocess_base.drop_nan(test_data)
    test_input = torch.tensor(test_input.loc[:,['id','height','width','aratio','local']].values).float()

    train_dataset = preprocess_base.tensor_dataset(train_input, train_label)

    joblib.dump(train_dataset,'../../data/advitisement/train')
    joblib.dump(test_input,'../../data/advitisement/test')

    utils.update_json('./hyper_parameters.json', {'input_dim':input_dim})
 
if __name__=='__main__':
    preprocess()