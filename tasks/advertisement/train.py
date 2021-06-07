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
    def __init__(self, model, optim, criterion, epochs, dataset, batch_size):
        super().__init__(model, optim, criterion, epochs, dataset, batch_size)

    def train(self):
        for epoch in range(self.epochs):
            self.kfold_cross_validation(self.dataset, self.batch_size, epoch, self.study)


def main():
    with open('./hyper_parameters.json') as f:
        model_param = json.load(f)

    dataset = joblib.load('../../data/advitisement/train')

    model = Classifier(model_param['input_dim'], model_param['hidden_dim'], model_param['target_dim'])
    optimizer = optim.SGD(model.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = 5000
    batch_size = 8

    do_train = Train(model, optimizer, criterion, epochs, dataset, batch_size)
    do_train.train()

    do_train.save_weight('./weight')

if __name__=='__main__':
    main()