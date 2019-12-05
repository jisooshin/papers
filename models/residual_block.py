#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
  def __init__(self, upper_lambda=1, p=2):
    super(ResBlock, self).__init__()
    self.conv_origin = nn.Conv2d(3, 64, (3, 3), padding=1)
    # weight constraint
    self.conv_origin.weight = nn.Parameter(
      self.conv_origin.weight / \
        (torch.norm(self.conv_origin.weight, p=p)) * upper_lambda,
      requires_grad=True)
    self.conv1 = nn.Conv2d(64, 64, (3, 3), padding=1)
    self.conv1.weight = nn.Parameter(
      self.conv1.weight / \
        (torch.norm(self.conv1.weight, p=p)) * upper_lambda,
      requires_grad=True)
    self.conv2 = nn.Conv2d(64, 64, (3, 3), padding=1)
    self.conv2.weight = nn.Parameter(
      self.conv2.weight / \
        (torch.norm(self.conv2.weight, p=p)) * upper_lambda,
      requires_grad=True)
    self.batchnorm1 = nn.BatchNorm2d(num_features=64)
    self.batchnorm2 = nn.BatchNorm2d(num_features=64)
  
  def forward(self, x):
    if x.shape[1] == 3:
      origin_x = self.conv_origin(x)
      x = self.conv1(origin_x)
      x = self.batchnorm1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = self.batchnorm2(x)
      x = F.relu(x)
      x = torch.add(origin_x, x)
    elif x.shape[1] == 64:
      origin_x = x
      x = self.conv1(x)
      x = self.batchnorm1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = self.batchnorm2(x)
      x = F.relu(x)
      x = torch.add(origin_x, x)
    return x
    