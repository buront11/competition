import torch
import torch.nn as nn
import torch.optim as optim

import optuna

class TuneBase():
    def __init__(self):
        pass

    def objective(trial):

        return

    def tune(self):
        study = optuna.create_study()