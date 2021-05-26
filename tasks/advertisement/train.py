from torch.utils.data.dataset import Dataset
import tqdm
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import json
import sys

sys.path.append('../../')

from models.classifier import Classifier
from train_base import ClassifierTrain

class Train(ClassifierTrain):
    def __init__(self, model, optim, criterion, epochs, dataset, batch_size, train_input, train_label):
        super().__init__(model, optim, criterion, epochs, dataset, batch_size)
        self.train_input = train_input
        self.train_label = train_label

    def train(self):
        for epoch in range(self.epochs):
            self.stratified_kfold(self.dataset, self.train_input, self.train_label, self.batch_size, epoch, self.study)


def main():
    with open('./hyper_parameters.json') as f:
        model_param = json.load(f)

    dataset = joblib.load('../../data/advitisement/train')
    split_data = joblib.load('../../data/advitisement/train_split')
    input_data = split_data[0]
    label_data = split_data[1]

    model = Classifier(model_param['input_dim'], model_param['hidden_dim'], model_param['target_dim'])
    optimizer = optim.SGD(model.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = 100
    batch_size = 10

    do_train = Train(model, optimizer, criterion, epochs, dataset, batch_size, input_data, label_data)
    do_train.train()

    do_train.save_weight('./weight')

if __name__=='__main__':
    main()