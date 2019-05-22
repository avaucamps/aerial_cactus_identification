from Training import Training
import torch.nn as nn


if __name__ == '__main__':
    train = Training(
        n_epochs=5, 
        batch_size=256
    )
    train.train(
        criterion=nn.CrossEntropyLoss(), 
        learning_rate=0.001
    )