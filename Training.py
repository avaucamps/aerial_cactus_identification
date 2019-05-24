from __future__ import print_function, division
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import utils, models
from tqdm import tqdm
from utils import get_device
from dataset.DatasetHelper import DatasetHelper
from torch.optim import lr_scheduler
from model.CNN import CNN
from ModelType import ModelType
import copy

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Training:
    def __init__(self):
        self.device = get_device()


    def train(self, n_epochs, batch_size, model_type, criterion, learning_rate, enable_scheduler=False):
        """
        Args:
        - n_epochs: number of epochs to train the neural network for.
        - batch_size: batch_size to use to train the neural network.
        - model_type: choice of ModelType enum.
        - criterion: PyTorch criterion to use.
        - learning_rate: learning rate to use to optimize the network.
        - enable_scheduler: whether to decay learning rate during the training.
        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_loader, self.validation_loader = self._load_dataset()
        self.model = self._load_model(model_type)

        self.criterion = criterion
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        best_accuracy = 0.0
        for epoch in range(self.n_epochs):
            print("---------- Epoch [{}/{}] ----------".format(str(epoch+1), str(self.n_epochs)))

            if enable_scheduler:
                scheduler.step()
            training_loss, training_accuracy = self._training_pass()
            print('Loss: {:.4f} Acc: {:.4f}'.format(training_loss, training_accuracy))

            validation_loss, validation_accuracy = self._validation_pass()
            print('Loss: {:.4f} Acc: {:.4f}'.format(validation_loss, validation_accuracy))

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_model_wts = copy.deepcopy(self.model.state_dict())

        print("Finished training")


    def _load_dataset(self):
        dataset_helper = DatasetHelper(self.batch_size)
        train_loader, validation_loader = dataset_helper.load_dataset()

        return train_loader, validation_loader


    def _load_model(self, model_type):
        if model_type == ModelType.cnn:
            return CNN().to(self.device)
        elif model_type == ModelType.fine_tuned:
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
            return model.to(self.device)
        elif model_type == ModelType.fixed_feature_extractor:
            pass
        else:
            pass

    
    def _training_pass(self):
        print("Training pass ...")
        model = self.model.train()
        with torch.set_grad_enabled(True):
            running_loss = 0
            correct = 0
            total = 0

            for i, data in enumerate(tqdm(self.train_loader, 0)):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        loss = running_loss / len(self.train_loader)
        accuracy = (100 * correct / total)
        return loss, accuracy


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