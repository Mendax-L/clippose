import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Depth_Net(nn.Module):
    def __init__(self):
        super(Depth_Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(18, 1)
            )
      
    def forward(self, rotation):
        t = self.fc(rotation)
        t = torch.sigmoid(t)
        return t.flatten()
    
