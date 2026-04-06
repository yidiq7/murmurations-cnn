"""1D CNN for predicting the analytic rank of elliptic curves."""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """1D CNN from Section 5 of the paper.

    Architecture: 3 Conv1d layers (16, 32, 64 channels, kernel_size=3, padding=1)
    each followed by ReLU and MaxPool1d(kernel_size=2, padding=1), then dropout(0.5)
    and two fully connected layers (128 units each) before the output layer.
    """

    def __init__(self, input_length, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, padding=1)

        # Compute flattened size via dummy forward pass
        dummy = torch.zeros(1, 1, input_length)
        self.flattened_size = self._get_flattened_size(dummy)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _get_flattened_size(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
