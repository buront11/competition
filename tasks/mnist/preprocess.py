import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import sys
import joblib

sys.path.append('../../')

import utils

def preprocess():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])

    trainset = torchvision.datasets.MNIST(root='../../data/', 
                                            train=True,
                                            download=True,
                                            transform=transform)

    testset = torchvision.datasets.MNIST(root='../../data/', 
                                            train=False, 
                                            download=True, 
                                            transform=transform)

    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    joblib.dump(trainset, './../../data/MNIST/train')
    joblib.dump(testset, './../../data/MNIST/test')

if __name__=='__main__':
    preprocess()