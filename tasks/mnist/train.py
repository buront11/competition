import torch
import torch.nn as nn
import torch.optim as optim

import joblib
import sys
import json

from torch.utils import data
sys.path.append('../../')

from train_base import ClassifierTrain

from models.cnn import MnistCNN

def main():
    with open('./hyper_parameters.json') as f:
        model_param = json.load(f)

    dataset = joblib.load('./../../data/MNIST/train')

    model = MnistCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train = ClassifierTrain(model=model, optim=optimizer, criterion=criterion, epochs=30, dataset=dataset, batch_size=100)
    train.train()

    train.save_weight('.')

if __name__=='__main__':
    main()