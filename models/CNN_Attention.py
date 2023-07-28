import torch
from torch import nn

class CNN_Attention(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding='valid'),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), padding='valid'),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 1)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), padding='valid'),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 1)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), padding='valid'),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 1)
        )
        
        # this shouldn't be here according to paper
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), padding='valid'),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 1)
        )
            
        self.ffn1 = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
        )
        
        self.attention = nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)
                
        self.ffn1 = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
        )
        
        self.aap = nn.AdaptiveAvgPool1d(1)
        
        self.ffn2 = nn.Sequential(
            nn.LazyLinear(out_classes),
            nn.Softmax(dim=-1)  # shoudn't be here with CrossEntropyLoss
        )
        
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        batch_size, ch, freq_dim, time_dim = x.shape
        
        x = torch.reshape(x, (batch_size, ch * freq_dim, time_dim))
        # x = torch.reshape(x, (batch_size, ch, time_dim * freq_dim))  # should be tested
        # x = torch.reshape(x, (batch_size, ch *time_dim, freq_dim))  # should be tested
        
        x = self.ffn1(x)
        
        x, _ = self.attention(x, x, x)
                
        # x = self.aap(x)
        # x = torch.squeeze(x, dim=-1)
        x = torch.mean(x, -1)
        
        x = self.ffn2(x)
                
        return x
    

if __name__ == "__main__":
    c_a = CNN_Attention()
    out = c_a(torch.rand(1, 1, 40, 67))
    print(out)