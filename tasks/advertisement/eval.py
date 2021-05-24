import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.modules import module
import torch.optim as optim
from torch.utils.data import DataLoader

import json
import sys
sys.path.append('../../')

from eval_base import EvalBase
from models.classifier import Classifier

class Eval(EvalBase):
    def __init__(self, model, model_path, dataset):
        super().__init__(model, model_path)
        self.dataset = dataset
        self.test_dataloader = DataLoader(self.dataset)

    def eval(self):
        result = []
        for data in self.test_dataloader:
            index, input = data
            output = self.model(input)

            _, predicted = torch.max(output.data, 1)
            result.append([index.item(),predicted.item()])

        index, value = zip(*result)
        df = pd.Series(value, index=index)
        print(df)

        df.to_csv('submit.csv')

def main():
    with open('./hyper_parameters.json') as f:
        model_param = json.load(f)

    dataset = joblib.load('../../data/advitisement/test')

    do_eval = Eval(Classifier(model_param['input_dim'], model_param['hidden_dim'], model_param['target_dim']),'weight', dataset)
    do_eval.eval()

if __name__=='__main__':
    main()
