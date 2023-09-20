"""
About model: https://arxiv.org/pdf/2105.07302.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet1D(nn.Module):
    def __init__(self, out_classes=3, **kwargs):
        super().__init__()
        self.stride = kwargs["stride"]

        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, stride=3)
        self.block1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256)
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256)
        )
        self.block5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256)
        )
        self.block6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256)
        )
        self.block7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256)
        )
        # self.block8 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.25),
        #     nn.Conv1d(256, 512, kernel_size=3, stride=1, padding='same'),
        #     nn.BatchNorm1d(512)
        # )
        # self.block9 = nn.Sequential(
        #     nn.Conv1d(512, 512, kernel_size=3, stride=1, padding='same'),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.25),
        #     nn.Conv1d(512, 512, kernel_size=3, stride=1, padding='same'),
        #     nn.BatchNorm1d(512)
        # )

        self.conv2 = nn.Conv1d(256, 256, kernel_size=1, stride=1)

        self.fc = nn.LazyLinear(out_classes)

    def forward(self, x):
        x = self.conv1(x)

        res_b = self.block1(x)
        x = res_b + x
        x = F.leaky_relu(x)
        x = F.max_pool1d(x, 3, 3)

        x = self.block2(x)
        # x = res_b + x
        x = F.leaky_relu(x)
        x = F.max_pool1d(x, 3, 3)

        res_b = self.block3(x)
        x = res_b + x
        x = F.leaky_relu(x)
        x = F.max_pool1d(x, 3, 3)

        res_b = self.block4(x)
        x = res_b + x
        x = F.leaky_relu(x)
        x = F.max_pool1d(x, 3, 3)

        res_b = self.block5(x)
        x = res_b + x
        x = F.leaky_relu(x)
        x = F.max_pool1d(x, 3, 3)

        res_b = self.block6(x)
        x = res_b + x
        x = F.leaky_relu(x)
        x = F.max_pool1d(x, 3, 3)

        res_b = self.block7(x)
        x = res_b + x
        x = F.leaky_relu(x)
        x = F.max_pool1d(x, 3, 3)

        # x = self.block8(x)
        # # x = res_b + x
        # x = F.leaky_relu(x)
        # x = F.max_pool1d(x, 3, 3)

        # res_b = self.block9(x)
        # x = res_b + x
        # x = F.leaky_relu(x)
        # x = F.max_pool1d(x, 3, 3)

        x = self.conv2(x)

        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.squeeze(x, -1)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    m5 = ResNet1D(stride=16, n_channel=40)
    out = m5(torch.rand(1, 1, 16000))
    print(out)
