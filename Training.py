from __future__ import print_function, division
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import utils, models
from tqdm import tqdm
from utils import get_device
from dataset.DatasetHelper import DatasetHelper
from model.CNN import CNN

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Training:
    def __init__(self, n_epochs, batch_size):
        """
        Args:
        - n_epochs: number of epochs to train the neural network for.
        - batch_size: batch_size to use to train the neural network.
        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = get_device()


    def train(self, criterion, learning_rate):
        """
        Args:
        - criterion: PyTorch criterion to use.
        - learning_rate: learning rate to use to optimize the network.
        """
        self.train_loader, self.validation_loader = self._load_dataset()
        self.model = self._load_model()

        self.criterion = criterion
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

        for epoch in range(self.n_epochs):
            print("---------- Epoch [{}/{}] ----------".format(str(epoch+1), str(self.n_epochs)))

            training_loss = self._training_pass()
            print('Training loss: {:.4f}'.format(training_loss))

            validation_loss, validation_accuracy = self._validation_pass()
            print('Validation loss: {:.4f}'.format(validation_loss))
            print('Validation accuracy: {:.4f}'.format(validation_accuracy))
            
        print("Finished training")


    def _load_dataset(self):
        dataset_helper = DatasetHelper(self.batch_size)
        train_loader, validation_loader = dataset_helper.load_dataset()

        return train_loader, validation_loader


    def _load_model(self):
        return CNN().to(self.device)

    
    def _training_pass(self):
        print("Training pass ...")
        model = self.model.train()
        running_loss = 0
        for i, data in enumerate(tqdm(self.train_loader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)


    def _validation_pass(self):
        print("Validation pass ...")
        self.model = self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            valid_loss = 0

            for i, data in enumerate(tqdm(self.validation_loader, 0)):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                valid_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss = valid_loss / len(self.validation_loader)
        accuracy = (100 * correct / total)
        return loss, accuracy