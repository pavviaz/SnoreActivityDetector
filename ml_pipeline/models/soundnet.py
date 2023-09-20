import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundNet(nn.Module):
    def __init__(self, out_classes=2, **kwargs):
        super().__init__()
        self.enc_1 = nn.Sequential(
            # change kernel_size depending on sr (80 for 8000 and so on)
            # n_channel = 32 as default
            nn.Conv1d(1, 16, kernel_size=16, stride=8),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.enc_2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=8, stride=2),
        )
        self.enc_3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=16, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.enc_4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.enc_5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=4, stride=2),
        )
        self.ff_1 = nn.Sequential(
            nn.LazyLinear(128),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.ff_2 = nn.Sequential(
            nn.Linear(128, 64),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fc = nn.Linear(64, out_classes)

    def forward(self, x):
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)

        x = torch.flatten(x, start_dim=1)

        x = self.ff_1(x)
        x = self.ff_2(x)

        x = self.fc(x)

        return F.softmax(x, dim=-1)


if __name__ == "__main__":
    m5 = SoundNet()
    out = m5(torch.rand(1, 1, 32000))
    print(out)
