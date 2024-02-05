import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import numpy

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        #Conv2d(channels, output_channels, (kernel_height, kernel_width), stride, padding)
        #1*32*32->6*28*28
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5)),
            nn.ReLU()
        )
        #6*28*28->6*14*14
        self.s2 = nn.Sequential(
            nn.MaxPool2d((2, 2), padding=0)
        )
        #6*14*14->16*10*10
        self.c3 = nn.Sequential(
            nn.Conv2d(6, 16, (5, 5), stride=1, padding=0),
            nn.ReLU()
        )
        #16*10*10->16*5*5
        self.s4 = nn.Sequential(
            nn.MaxPool2d((2, 2), padding=0)
        )
        #16*5*5->120
        self.F5 = nn.Sequential(
            nn.Linear(5*5*16, 120),
            nn.ReLU()
        )
        self.F6 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.F7 = nn.Sequential(
            nn.Linear(84, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.c1(x)
        # print(x.shape)
        x = self.s2(x)
        # print(x.shape)
        x = self.c3(x)
        # print(x.shape)
        x = self.s4(x)
        # print(x.shape)
        x = x.view(-1, 5*5*16)
        x = self.F5(x)
        # print(x.shape)
        x = self.F6(x)
        # print(x.shape)
        x = self.F7(x)
        # print(x.shape)
        return x

if __name__ == "__main__":
    inp = torch.randn(1, 1, 32, 32)
    ynet = Lenet()
    out = ynet(inp)
    print(out)