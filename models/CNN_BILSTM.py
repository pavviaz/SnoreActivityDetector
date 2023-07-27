import torch
from torch import nn
from torch.nn import functional as F
from models.lstm_sigmoid import LSTMSigmoid


class CNN_BILSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, (5, 5), padding='valid')
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding='valid')
        
        self.fc1 = nn.Linear(3584, 64)
        
        self.lstm = LSTMSigmoid(64, 32, batch_first=True, bidirectional=True)
        
        self.fc2 = nn.Linear(64, 2)
            
        
    def forward(self, x):
        features_block1 = torch.max_pool2d(F.mish(self.conv1(x)), kernel_size=(2, 2))
        
        features_block2 = torch.max_pool2d(input=F.mish(self.conv2(features_block1)), kernel_size=(2, 2))
        
        ff1 = torch.reshape(torch.relu(self.fc1(torch.flatten(input=torch.permute(features_block2, (0, 2, 3, 1)), start_dim=1))), (-1, 1, 64))
        
        _, (hidden_state, _) = self.lstm(ff1)

        conc_hidden = torch.cat([hidden_state[0], hidden_state[1]], dim=-1)
        
        out = torch.softmax(self.fc2(conc_hidden), dim=-1)
        
        return out


if __name__ == "__main__":
    net = CNN_BILSTM()
    x = torch.randn((1, 1, 40, 67))
    net(x)
