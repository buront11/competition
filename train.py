import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import LSTMPrediction

import utils

def train():
    train_dataset, test_data = utils.get_flights_dataset()

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,drop_last = True)

    model = LSTMPrediction(1, 50, 1, 2)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 150

    for i in range(epochs):
        for seq, labels in train_dataloader:
            optimizer.zero_grad()

            y_pred, _ = model(seq)

            batch_loss = loss_function(y_pred, labels)
            batch_loss.backward()
            optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {batch_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {batch_loss.item():10.10f}')

    test_dataset = train_dataset[-12:]
    test_dataloader = DataLoader(test_dataset, batch_size=2)

    for i in range(12):
        seq = torch.FloatTensor(test_dataset[-12:])
        with torch.no_grad():
            test_dataset.append(model(seq).item())


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    actual_predictions = scaler.inverse_transform(np.array(test_dataset[12:] ).reshape(-1, 1))
    print("pred:{}, label:{}".format(actual_predictions, test_data))


if __name__=='__main__':
    train()