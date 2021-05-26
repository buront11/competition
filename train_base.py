import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold, StratifiedKFold

import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from tqdm.std import TqdmDeprecationWarning

from utils import EarlyStopping

class TrainBase():
    def __init__(self, model, optim, criterion, epochs):
        self.model = model
        self.optimizer = optim
        self.criterion = criterion
        self.epochs = epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.train_loss = 0
        self.valid_loss = 0
        self.train_total = 0
        self.valid_total = 0
        self.train_correct = 0
        self.valid_correct = 0

    def kfold_cross_validation(self, dataset, batch_size, epoch, study_func, n_splits=4):
        kf = KFold(n_splits=n_splits)

        for _fold, (train_index, valid_index) in enumerate(kf.split(dataset)):

            train_dataset = Subset(dataset, train_index)
            train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
            valid_dataset   = Subset(dataset, valid_index)
            valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)

            self.init_param()
            
            # そのうちtrainとvalidで別関数化する
            study_func(train_dataloader, valid_dataloader)

            self.train_loss += self.train_loss / kf.n_splits
            self.train_correct += self.train_correct / kf.n_splits
            self.train_total += self.train_total / kf.n_splits
            self.valid_loss += self.valid_loss / kf.n_splits
            self.valid_correct += self.valid_correct / kf.n_splits
            self.valid_total += self.valid_total / kf.n_splits

        print("epoch : [{}/{}]".format(epoch+1, self.epochs))
        print("train loss: {}".format(self.train_loss))
        print("train acc : {}".format(self.train_correct/self.train_total))
        print("valid loss: {}".format(self.valid_loss))
        print("valid acc : {}".format(self.valid_correct/self.valid_total))

    def stratified_kfold(self, dataset, input_data, label_data, batch_size, epoch, study_func, n_splits=2):
        skf = StratifiedKFold(n_splits=n_splits)

        for _fold, (train_index, valid_index) in enumerate(skf.split(input_data, label_data)):

            train_dataset = Subset(dataset, train_index)
            train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
            valid_dataset   = Subset(dataset, valid_index)
            valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)

            self.init_param()
            
            # そのうちtrainとvalidで別関数化する
            study_func(train_dataloader, valid_dataloader)

            self.train_loss += self.train_loss / skf.n_splits
            self.train_correct += self.train_correct / skf.n_splits
            self.train_total += self.train_total / skf.n_splits
            self.valid_loss += self.valid_loss / skf.n_splits
            self.valid_correct += self.valid_correct / skf.n_splits
            self.valid_total += self.valid_total / skf.n_splits

        print("epoch : [{}/{}]".format(epoch+1, self.epochs))
        print("train loss: {}".format(self.train_loss))
        print("train acc : {}".format(self.train_correct/self.train_total))
        print("valid loss: {}".format(self.valid_loss))
        print("valid acc : {}".format(self.valid_correct/self.valid_total))

    def init_param(self):
        self.train_loss = 0
        self.valid_loss = 0
        self.train_total = 0
        self.valid_total = 0
        self.train_correct = 0
        self.valid_correct = 0

    def save_weight(self, save_path):
        torch.save(self.model.state_dict(), save_path)

class ClassifierTrain(TrainBase):
    def __init__(self, model, optim, criterion, epochs, dataset, batch_size):
        super().__init__(model, optim, criterion, epochs)
        self.dataset = dataset
        self.batch_size = batch_size

    def train(self, early_stopping=None):
        """訓練するmethod

        Parameters
        ----------
        early_stopping : instance, optional
            Early stoppingを行う際はEarlyStoppingのinstanceを渡す, by default None
        """
        for epoch in range(self.epochs):
            self.kfold_cross_validation(self.dataset, self.batch_size, epoch, self.study)

            if early_stopping is not None:
                early_stopping(self.train_loss, self.model)
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    break

    def study(self, train_dataloader, valid_dataloader):

        self.model.train()
        for input, labels in train_dataloader:
            input, labels = input.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(input)

            _, predicted = torch.max(output.data, 1)
            self.train_total += labels.size(0)
            self.train_correct += (predicted == labels).sum().item()

            train_loss = self.criterion(output, labels)
            train_loss.backward()
            self.optimizer.step()

            self.train_loss += train_loss.item()

        self.model.eval()
        with torch.no_grad():
            for input, labels in valid_dataloader:
                input, labels = input.to(self.device), labels.to(self.device)
                output = self.model(input)

                _, predicted = torch.max(output.data, 1)
                self.valid_total += labels.size(0)
                self.valid_correct += (predicted == labels).sum().item()

                valid_loss = self.criterion(output, labels)

                self.valid_loss += valid_loss.item()