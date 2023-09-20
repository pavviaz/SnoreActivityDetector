"""
About M5 model: https://arxiv.org/pdf/1610.00087.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class M5Net(nn.Module):
    """
    M5 CNN model with sensitive input filters
    for RAW Waveform primarily (1D-data)
    """

    def __init__(self, out_classes=3, **kwargs):
        super().__init__()
        self.stride = kwargs["stride"]
        self.n_channel = kwargs["n_channel"]

        self.conv1 = nn.Sequential(
            # change kernel_size depending on sr (80 for 8000 and so on)
            # n_channel = 32 as default
            nn.Conv1d(1, self.n_channel, kernel_size=160, stride=self.stride),
            nn.BatchNorm1d(self.n_channel),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.n_channel, self.n_channel, kernel_size=3),
            nn.BatchNorm1d(self.n_channel),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(4),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.n_channel, 2 * self.n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * self.n_channel),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(4),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(2 * self.n_channel, 2 * self.n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * self.n_channel),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(4),
        )

        self.fc = nn.LazyLinear(out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.squeeze(x, -1)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    m5 = M5Net(stride=16, n_channel=40)
    out = m5(torch.rand(1, 1, 32000))
    print(out)
