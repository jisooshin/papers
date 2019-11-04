#-*- coding:utf-8 -*-
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F


class ResNet34(nn.Module):
  """
  input tensor shape. (batch, 3, 224, 224)
  output logit shape. (batch, 100)
  """
  def __init__(self):
    super(ResNet34, self).__init__()
    self.conv1 = nn.Conv2d(
      in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2)
    self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
    
    self.idt2 = nn.Identity()
    self.conv2list = [
      nn.Conv2d(64, 64, (3, 3), padding=1) for x in range(2)]
    
    self.conv2to3 = nn.Conv2d(64, 128, (3, 3), padding=1)
    
    self.idt3 = nn.Identity()
    self.conv3list = [
      nn.Conv2d(128, 128, (3, 3), padding=1) for x in range(2)]
    self.conv3to4 = nn.Conv2d(128, 256, (3, 3), padding=1)
    
    self.idt4 = nn.Identity()
    self.conv4list = [
      nn.Conv2d(256, 256, (3, 3), padding=1) for x in range(2)]
    self.conv4to5 = nn.Conv2d(256, 512, (3, 3))
    
    self.idt5 = nn.Identity()
    self.conv5list = [
      nn.Conv2d(512, 512, (3, 3), padding=1) for x in range(2)]

    self.batchnorm1 = nn.BatchNorm2d(num_features=64)
    self.batchnorm2 = nn.BatchNorm2d(num_features=128)
    self.batchnorm3 = nn.BatchNorm2d(num_features=256)
    self.batchnorm4 = nn.BatchNorm2d(num_features=512)
    
    self.globalavgpool = nn.AvgPool2d(kernel_size=(52, 52)) 
    self.fc = nn.Linear(512, 100)
    self.softmax = nn.Softmax()
    
  def forward(self, x):
    def _list_to_layer(x, layer_list, batchnorm_obj):
      for val in layer_list:
        x = val(x)
        x = batchnorm_obj(x)
        x = F.relu(x)
      return x

    x = self.conv1(x)
    x = F.relu(self.batchnorm1(x))
    x = self.maxpool(x)

    idt_2_1 = self.idt2(x)
    x = _list_to_layer(x, self.conv2list, self.batchnorm1)
    x = idt_2_1 + x
    idt_2_2 = self.idt2(x)
    x = _list_to_layer(x, self.conv2list, self.batchnorm1)
    x = idt_2_2 + x
    idt_2_3 = self.idt2(x)
    x = _list_to_layer(x, self.conv2list, self.batchnorm1)
    x = idt_2_3 + x
    x = self.conv2to3(x)
    
    idt_3_1 = self.idt3(x)
    x = _list_to_layer(x, self.conv3list, self.batchnorm2)
    x = idt_3_1 + x
    idt_3_2 = self.idt3(x)
    x = _list_to_layer(x, self.conv3list, self.batchnorm2)
    x = idt_3_2 + x
    idt_3_3 = self.idt3(x)
    x = _list_to_layer(x, self.conv3list, self.batchnorm2)
    x = idt_3_3 + x
    idt_3_4 = self.idt3(x)
    x = _list_to_layer(x, self.conv3list, self.batchnorm2)
    x = idt_3_4 + x
    x = self.conv3to4(x)
    
    idt_4_1 = self.idt4(x)
    x = _list_to_layer(x, self.conv4list, self.batchnorm3)
    x = idt_4_1 + x
    idt_4_2 = self.idt4(x)
    x = _list_to_layer(x, self.conv4list, self.batchnorm3)
    x = idt_4_2 + x
    idt_4_3 = self.idt4(x)
    x = _list_to_layer(x, self.conv4list, self.batchnorm3)
    x = idt_4_3 + x
    idt_4_4 = self.idt4(x)
    x = _list_to_layer(x, self.conv4list, self.batchnorm3)
    x = idt_4_4 + x
    idt_4_5 = self.idt4(x)
    x = _list_to_layer(x, self.conv4list, self.batchnorm3)
    x = idt_4_5 + x
    idt_4_6 = self.idt4(x)
    x = _list_to_layer(x, self.conv4list, self.batchnorm3)
    x = idt_4_6 + x
    x = self.conv4to5(x)
    
    idt_5_1 = self.idt5(x)
    x = _list_to_layer(x, self.conv5list, self.batchnorm4)
    x = idt_5_1 + x
    idt_5_2 = self.idt5(x)
    x = _list_to_layer(x, self.conv5list, self.batchnorm4)
    x = idt_5_2 + x
    idt_5_3 = self.idt5(x)
    x = _list_to_layer(x, self.conv5list, self.batchnorm4)
    x = idt_5_3 + x
    
    x = self.globalavgpool(x).view([-1, 512])
    x = self.fc(x)
    
    return x