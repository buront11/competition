import torch
import torch.nn as nn

class LSTMPrediction(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(LSTMPrediction,self).__init__()
        self.feature_dim = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size,hidden_size, num_layers=1, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self,inputs, hidden_cell=None):
        # inputを[batch size, seq_len, feature dim]の3次元データに変換
        lstm_out, (hidden, cell) = self.lstm(inputs.view(self.batch_size, inputs.size(1), self.feature_dim), hidden_cell)
        outputs = self.linear(lstm_out[:,-1,:])
        return outputs, (hidden, cell)

