from Training import Training
import torch.nn as nn
from ModelType import ModelType

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from utils import get_device
from dataset.DatasetHelper import DatasetHelper


if __name__ == '__main__':
    train = Training()

    train.train(
        n_epochs=5,
        batch_size=64,
        model_type=ModelType.cnn,
        criterion=nn.CrossEntropyLoss(),
        learning_rate=0.001
    )

    train.train(
        n_epochs=5,
        batch_size=64,
        model_type=ModelType.fine_tuned,
        criterion=nn.CrossEntropyLoss(), 
        learning_rate=0.001,
        enable_scheduler=True
    )