#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlockOutput(nn.Module):
  def __init__(self):
    super(ResBlockOutput, self).__init__()
    self.globalavgpool = nn.AvgPool2d(kernel_size=(224, 224))
    self.fc1 = nn.Linear(3, 100)
    self.fc2 = nn.Linear(64, 100)
    
  def forward(self, x):
    if x.shape[1] == 3:
      x = self.globalavgpool(x).view(-1, 3)
      logit = self.fc1(x)
    elif x.shape[1] == 64:
      x = self.globalavgpool(x).view(-1, 64)
      logit = self.fc2(x)
    return logit