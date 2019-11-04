#-*- coding:utf-8 -*-
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
  
  def __init__(self):
    super(SimpleNN, self).__init__()


  def forward(self, x):
    return 