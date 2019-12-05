#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaNet(nn.Module):
  """
  Approximates Rademacher complexity as the square-root of the depth.
  Reference : https://bit.ly/33MjCg5
  """   
  def __init__(self, num_layers, module_instance, output_instance):
    super(AdaNet, self).__init__()
    self.NUM_LAYERS = num_layers
    module = [module_instance for i in range(1, num_layers)]
    output = [output_instance for i in range(num_layers)]
    weight = torch.ones(num_layers) / num_layers 
    self.weight = nn.Parameter(data=weight, requires_grad=True)
    
    self.modules_list = nn.ModuleList(module)
    self.outputs_list = nn.ModuleList(output) 
    self.softmax = nn.Softmax(dim=0)
    
  def forward(self, x):
    output = []
    for i in range(self.NUM_LAYERS):
      if i == 0:
        _output = self.outputs_list[0](x)        
        output.append(_output)
      else:
        x = self.modules_list[i - 1](x) 
        _output = self.outputs_list[i](x) 
        output.append(_output)
    output = torch.stack(output, dim=1)
    output = torch.matmul(self.softmax(self.weight), output)      
    rademacher_complexity = torch.sqrt(
      torch.tensor(self.NUM_LAYERS, dtype=torch.float32))
    return output, rademacher_complexity