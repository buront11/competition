# -*- coding: utf-8 -*-
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data
import numpy as np
import math
import scipy.stats
import random
import argparse
import xml.etree.ElementTree as ET
import joblib
import logging
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class BitCoin(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super(BitCoin,self).__init__()
        self.data = []  # データとなるテンソルのリスト
        self.label = []  # データに対応したラベル
        self.df = pd.read_csv('./data/bitcoin/train.csv')
        self.columns = ['Date','ETHUSD_Close_log','ZECUSD_Close_log','LTCUSD_Close_log','BTCUSD_Close_log']
        self.df = self.df[self.columns]
        print(self.df.info())

    def __getitem__(self,idx):
        pass

class AdvertisementDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(AdvertisementDataset, self).__init__()
        self.train_data = pd.read_csv('./data/advitisement/train.tsv', delimiter='\t')
        self.test_data = pd.read_csv('./data/advitisement/test.tsv', delimiter='\t')

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self,idx):
        pass
    
if __name__=='__main__':
    a = AdvertisementDataset()