import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # [bz, 32, 195, 610]
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # [bz, 64, 195, 610]
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # [bz, 128, 195, 610]

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 24 * 76, 1024)  # Adjust input size based on output of final conv layer
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        # Forward pass through conv layers
        x = self.pool(self.relu(self.conv1(x)))  # Output size: [bz, 32, 97, 305]
        x = self.pool(self.relu(self.conv2(x)))  # Output size: [bz, 64, 48, 152]
        x = self.pool(self.relu(self.conv3(x)))  # Output size: [bz, 128, 24, 76]
        
        # Flatten the output from the conv layers
        x = x.view(x.size(0), -1)  # Flatten to [bz, 128 * 24 * 76]
        
        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))  # [bz, 1024]
        x = self.fc2(x)  # [bz, 512]
        
        return x


class FullyConnectedBranch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedBranch, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
