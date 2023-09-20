import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechFeatureEmbedding(nn.Module):
    def __init__(self, num_hid=64):
        super().__init__()
        self.conv1 = nn.LazyConv1d(num_hid, 5, stride=2, padding="valid")
        self.conv2 = nn.Conv1d(num_hid, num_hid, 5, stride=2, padding="valid")
        self.conv3 = nn.Conv1d(num_hid, num_hid, 5, stride=2, padding="valid")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class CNN_T(nn.Module):
    def __init__(self, out_classes=3, **kwargs):
        super(CNN_T, self).__init__()

        self.cnn_enc = SpeechFeatureEmbedding()

        self.fc_1 = nn.LazyLinear(64)

        self.transformer_enc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=kwargs["transformer_d_model"],
                nhead=kwargs["nhead"],
                dim_feedforward=kwargs["transformer_dimforward"],
                dropout=kwargs["transformer_dropout"],
                batch_first=True,
            ),
            num_layers=kwargs["transformer_enc_num"],
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.LazyLinear(out_classes)

    def forward(self, input):
        input = torch.squeeze(input, 1)
        input = torch.permute(input, (0, -1, 1))

        x = self.cnn_enc(input)

        x = self.fc_1(x)

        x = self.transformer_enc(x)

        x = self.pool(x)
        x = torch.squeeze(x, dim=-1)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = CNN_T(
        4,
        transformer_d_model=64,
        nhead=4,
        transformer_dimforward=512,
        transformer_dropout=0.2,
        transformer_enc_num=3,
    )
    model(torch.zeros(size=(1, 40, 201)))

    print()
