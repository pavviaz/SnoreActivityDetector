import torch
from torch import nn
from torch.nn import functional as F
from .lstm_sigmoid import LSTMSigmoid


class CNN_BILSTM(nn.Module):
    def __init__(self, out_classes=3, **kwargs):
        super().__init__()
        self.ffn_dim = kwargs["ffn_dim"]

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5), padding="valid"),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), padding="valid"),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.fc1 = nn.Sequential(nn.LazyLinear(self.ffn_dim), nn.ReLU())

        self.lstm = LSTMSigmoid(
            self.ffn_dim, 
            kwargs["lstm_dim"], 
            batch_first=True, 
            bidirectional=True
        )

        self.fc = nn.Linear(kwargs["lstm_dim"] * 2, out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = torch.reshape(x, (-1, 1, self.ffn_dim))

        _, (hidden_state, _) = self.lstm(x)

        x = torch.cat([hidden_state[0], hidden_state[1]], dim=-1)

        out = self.fc(x)

        return out


if __name__ == "__main__":
    net = CNN_BILSTM(out_classes=3, ffn_dim=512, lstm_dim=256)
    x = torch.randn((1, 1, 40, 201))
    print(net(x).shape)
