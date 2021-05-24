import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim

class EvalBase():
    def __init__(self, model, model_path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
