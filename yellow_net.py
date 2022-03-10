import torch

from yellow_data import yellow_data
from torch import nn
import numpy as np

class Net_v1(nn.Module):
    def __init__(self):
        super(Net_v1,self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(300*300*3,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 85),
            nn.ReLU(),
            nn.Linear(85, 85),
            nn.ReLU(),
            nn.Linear(85, 40),
            nn.ReLU(),
            nn.Linear(40, 5),

        )

    def forward(self,x):
        return self.fc_layers(x)

class Net_v2(nn.Module):
    def __init__(self):
        super(Net_v2, self).__init__()
        self.Con_layers = nn.Sequential(
         nn.Conv2d(3,24,(3,3)),
         nn.ReLU(),
         nn.MaxPool2d(3),
         nn.Conv2d(24,48,(3,3)),
         nn.ReLU(),
         nn.MaxPool2d(3),
         nn.Conv2d(48, 48, (3, 3)),
         nn.ReLU(),
         nn.MaxPool2d(3),
         nn.Conv2d(48, 96, 3),
         nn.ReLU(),
         nn.Conv2d(96, 198, 3),
         nn.ReLU(),
         nn.Conv2d(198, 256, 3),
         nn.ReLU(),               #torch.Size([1, 256, 4, 4])

     )
        self.out_layer = nn.Sequential(
            nn.Linear(256*4*4,5),
            nn.Sigmoid()          #控制输出的5个值在0-1之间

    )
    def forward(self,x):
       Con_out=self.Con_layers(x)
       Con_out= Con_out.reshape(-1,256*4*4)   #n.v
       out = self.out_layer(Con_out)
       return out


if __name__ == '__main__':
    net= Net_v1()
    x = torch.randn(1,300*300*3)   #torch.Size([1, 5])
    y = net(x)
    print(y.shape)
    # net =Net_v2()
    # x = torch.randn(1,3,300,300)
    # y = net(x)
    # print(y.shape)          #torch.Size([1, 5])
