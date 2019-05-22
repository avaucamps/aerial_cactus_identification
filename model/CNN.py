import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1, self.batch_norm1 = self.block(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2, self.batch_norm2 = self.block(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.conv3, self.batch_norm3 = self.block(in_channels=64, out_channels=128, kernel_size=3, padding=2)
        self.conv4, self.batch_norm4 = self.block(in_channels=128, out_channels=256, kernel_size=3, padding=2)
        self.conv5, self.batch_norm5 = self.block(in_channels=256, out_channels=512, kernel_size=3, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * 1 * 1, 2)
   

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
        x = self.pool(F.relu(self.batch_norm5(self.conv5(x))))
        x = self.avg(x)
        x = x.view(-1, 512 * 1 * 1)
        x = self.fc(x)

        return x


    def block(self, in_channels, out_channels, kernel_size, padding):
        conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        batch_norm = nn.BatchNorm2d(out_channels)

        return conv, batch_norm