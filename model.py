# MINIPROJECT IN APPLIED STATISTICS
import torch
import torch.nn as nn


class Network(nn.module):
    def __init__(self, input_channels: int = 3, output_channels: int = 1, layer_size: int = 32):
        super().__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(input_channels, layer_size),
            nn.LeakyReLU(),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(input_channels, layer_size),
            nn.LeakyReLU(),
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(input_channels, layer_size),
            nn.LeakyReLU(),
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(input_channels, layer_size),
            nn.LeakyReLU(),
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(input_channels, layer_size),
            nn.LeakyReLU(),
        )
        self.hidden6 = nn.Sequential(
            nn.Linear(input_channels, layer_size),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Linear(layer_size, output_channels)

    def forward(self, data):
        x = self.hidden1(data)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        out = self.classifier(x)
        return out
