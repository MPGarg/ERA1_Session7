import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 26 RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 24 RF = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12 RF = 6

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 12 RF = 6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 10 RF = 10

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 8 RF = 14

        # OUTPUT BLOCK
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 8 RF = 14

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1 RF = 28

        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.dropout(self.convblock1(x))
        x = self.dropout(self.convblock2(x))
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.dropout(self.convblock4(x))
        x = self.dropout(self.convblock5(x))
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)