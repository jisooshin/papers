#-*- coding: utf-8 -*-
import os
import math
import copy
import argparse
from datetime import datetime

import numpy as np

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.distributions as dist

from torch.utils.tensorboard import SummaryWriter
from models import adanet, residual_block, residual_block_output

parser = argparse.ArgumentParser(description='Train AdaNet model.')
parser.add_argument(
  '--base_path', type=str, default=os.environ['HOME'] + '/cifar100/adanet',
  help='Base path for tensorboard events file and learned model parameters.')
parser.add_argument(
  '--train_batch', type=int,
  help='Batch size of the training data.', default=100)
parser.add_argument(
  '--val_batch', type=int,
  help='Batch size of the validation data.', default=100)
parser.add_argument(
  '--max_epoch', type=int, default=10, 
  help='Max epoch.')
parser.add_argument(
  '--max_iter', type=int, default=50,
  help='Max iteration.')
parser.add_argument(
  '--patience', type=int, default=100)
parser.add_argument(
  '--threshold', type=float, default=0.)
parser.add_argument(
  '--verbose', type=int, default=100)
parser.add_argument(
  '--name', type=str, default=datetime.strftime(datetime.now(), '%Y%m%d_%H'))

#Hyperparams
parser.add_argument(
  '--upper_lambda', type=float, default=1)
parser.add_argument(
  '--p', type=float, default=2)
parser.add_argument(
  '--lower_lambda', type=float, default=0.0001)
parser.add_argument(
  '--beta', type=float, default=0.0001)


ARGS = parser.parse_args()

if os.path.isdir(ARGS.base_path):
  pass
else:
  os.makedirs(ARGS.base_path)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

writer = SummaryWriter(log_dir=ARGS.base_path + '/runs')
transform = transforms.Compose(
    [transforms.Pad(padding=(2, 2, 2, 2)), 
     transforms.RandomCrop(size=32),
     torchvision.transforms.RandomHorizontalFlip(p=0.5),
     torchvision.transforms.Resize(size=[224, 224]),
     transforms.ToTensor(),
     transforms.Normalize(
       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

train = torchvision.datasets.CIFAR100(
  root=ARGS.base_path + '/data', train=True, download=True, transform=transform) 
test = torchvision.datasets.CIFAR100(
  root=ARGS.base_path + '/data', train=False, download=True, transform=transform) 

trainlist = torch.utils.data.random_split(train, [40000, 10000])
train, val = trainlist[0], trainlist[1]

trainloader = torch.utils.data.DataLoader(
  train, batch_size=ARGS.train_batch, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(
  val, batch_size=ARGS.val_batch, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
  test, batch_size=ARGS.val_batch, shuffle=True, num_workers=2)


# objective 
def criterion(trained_logits, labels, 
              mode, penalize=torch.tensor(0),
              lambda_ = 0.0001, beta = 0.0001,
              weight=torch.tensor(0), training_logits=None,
              device=torch.device('cpu')):
  trained_logtis = trained_logits.to(device)
  labels = labels.to(device)
  penalize = penalize.to(device)
  penalize_term =  torch.mul(
    (lambda_ * penalize + beta),
    torch.abs(nn.Softmax(dim=0)(weight)))
  penalize_term = torch.sum(penalize_term)
  if mode == "train":
    training_logits = training_logits.to(device) 
    y_f = torch.mul(labels, trained_logits)
    y_wu = torch.mul(labels, training_logits)
    
    penalize_term = torch.sum(penalize_term)
    # surrogate loss
    loss = torch.log(torch.tensor(1.) + torch.exp(1 - y_f - y_wu))
    loss = torch.mean(loss) + penalize_term 

  elif mode == "eval":
    y_f = torch.mul(labels, trained_logits)
    loss = torch.log(torch.tensor(1.) + torch.exp(1 - y_f))
    loss = torch.mean(loss) + penalize_term
  else:
    raise Exception("Putting the right 'mode' argument.")
  return loss


base_module = residual_block.ResBlock(
  upper_lambda=ARGS.upper_lambda, p=ARGS.p)
out_module = residual_block_output.ResBlockOutput()

for t in range(1, ARGS.max_iter + 1):
  if t > 1: 
    ckpt_path = base_path + "/{}_checkpoint.pt".format(t - 1)
    checkpoint = torch.load(ckpt_path)
    print("Load {}".format(ckpt_path))
    
  h = adanet.AdaNet(t, base_module, out_module)
  h_prime = adanet.AdaNet(t + 1, base_module, out_module)
  
  h = h.to(device)
  h_prime = h_prime.to(device)
  weaklearners = [h, h_prime]
  
  min_objective = {"h": 0., "h_prime": 0.}
  NUM_CLASSES = 100
  for w in range(2):
    weaklearner = weaklearners[w]
    optimizer = optim.Adam(params=weaklearner.parameters())
    current_w = "h" if w == 0 else "h_prime"
    print("#------------------------------------------------------------#")
    print("Start {}".format(current_w))
    early_metrics = []
    min_value = 0.
    patience = 0
    global_steps = 0
    for epoch in range(ARGS.max_epoch):
      running_objective = 0.
      steps_per_epoch = 1.
      verbose_loss = 0.
      for i, data in enumerate(trainloader):
        global_steps += 1
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        _label = torch.zeros(
          [ARGS.train_batch, NUM_CLASSES], dtype=torch.float32)
        _label[range(_label.shape[0]), labels] = 1
        labels = _label
        optimizer.zero_grad()
        
        if t == 1:
          prev_logit = torch.tensor(0.).to(device)
          prev_weight = torch.tensor(0.).to(device)
        else:
          prev_modelParams = [param for param in checkpoint['model_state_dict']]
          prev_modelWeight = {
            k: nn.Softmax(dim=0)(v) for k, v in checkpoint['model_state_dict'].items()
            if k == 'weight'}
          param_locate = [param for param in weaklearner.state_dict()]
          prev_paramIdx = [param_locate.index(i) for i in prev_modelParams]
          prev_paramIdx.remove(0)

          idx = 0
          for param in weaklearner.parameters():
            if idx in prev_paramIdx:
              param.requires_grad = False
            else:
              param.requires_grad = True
            idx += 1
          
          if checkpoint['h_or_hprime'] == 'h_prime':
            prev_f = AdaNet(t, base_module, out_module)
          else:
            prev_f = AdaNet(t - 1, base_module, out_module)
          prev_f = prev_f.to(device)
          # In minimize F(w, u) step, Have to use previous(trained) parameters.
          prev_f.load_state_dict(checkpoint['model_state_dict'], strict=False)
          prev_logit, _ = prev_f(inputs)
          prev_weight = checkpoint['model_state_dict']['weight']\
            .to(torch.device('cpu'))
          prev_weight = nn.Softmax(dim=0)(prev_weight)
          
        logits, rademacher_complexity = weaklearner(inputs)
        objective = criterion(
          trained_logits=prev_logit, weight=weaklearner.weight,
          training_logits=logits, penalize=rademacher_complexity,  
          labels=labels, mode="train", device=device)
        objective.backward(retain_graph=True)
        optimizer.step()
        
        steps_per_epoch += 1
        running_objective += objective.item()
        del inputs, labels, data, logits, _label
        
        verbose_loss += objective.item()
        if i % ARGS.verbose == ARGS.verbose - 1:
          if patience >= ARGS.patience:
            break
          #_metric = objective.item()
          _metric = verbose_loss / 100
          if len(early_metrics) < 2:
            early_metrics.append(_metric)
          elif len(early_metrics) >= 2:
            if _metric + ARGS.threshold >= min_value:
              patience += 1
            early_metrics.append(_metric)
            min_value = min(early_metrics)
            early_metrics.sort()
            early_metrics = early_metrics[:2]
        
          print("**** ITERATION [{}] ****".format(t))
          if w == 0: 
            print("Learning ** h **")
          else:
            print("Learning ** h_prime **")
          print(
            "EPOCH [{}] | GLOBAL STEP [{}]".format(epoch + 1, global_steps))
          print("Running Loss: {0:.8f}".format(verbose_loss / 100))
          print("Loss: {0:.8f}".format(verbose_loss / 100))
          print("Min Loss: {0:.8f}".format(min_value))
          print("Iter[{}] | w[{}]: Patience added: {}"\
                .format(t, current_w, patience))
          print("Trained weight", nn.Softmax(dim=0)(weaklearner.weight))
          print("------------------------------------------------")
          verbose_loss = 0.
        
          
    min_objective[current_w] = min_value
    print("#################################################################")
    print("Training end in global step {}".format(global_steps))
    print("Minimum objective: {0:.8f}".format(min_objective[current_w]))
    print("[{}] end.".format(current_w))
    print("#################################################################")

  print("Eval h and h_prime")
  if min_objective["h_prime"] >= min_objective["h"]:
    h_t = h
  else:
    h_t = h_prime
  
  h_t = h_t.to(torch.device('cpu'))
  weight_star = nn.Softmax(dim=0)(h_t.weight)
  print("weight_star", weight_star)
  if prev_weight == 0:
    weight_total = torch.add(weight_star, nn.Softmax(dim=0)(prev_weight))
  else:
    if weight_star.shape == prev_weight.shape:
      weight_total = torch.add(weight_star, nn.Softmax(dim=0)(prev_weight))
    else:
      zero_pad_size = weight_star.shape[0] - prev_weight.shape[0]
      weight_trained = F.pad(
        prev_weight, (0, zero_pad_size), 'constant', 0)
      print("Prvious weight", prev_weight)
      weight_total = torch.add(weight_star, nn.Softmax(dim=0)(prev_weight))
  print("weight total", weight_total, "weight prev", prev_weight)
  print("End combined weight gen.")

  val_i = 1
  val_total = 0.
  val_prev = 0.
  for val_i, val_data in enumerate(valloader):
    val_inputs, val_labels = val_data
    val_inputs = val_inputs.to(device)
    val_labels = val_labels.to(device)
    _label = torch.zeros([VAL_BATCH_SIZE, NUM_CLASSES], dtype=torch.float32)
    _label[range(_label.shape[0]), val_labels] = 1
    val_labels = _label

    case_a = copy.deepcopy(h_t) # weight_total
    case_b = copy.deepcopy(h_t) # previouse weight
    case_a = case_a.to(device)
    case_b = case_b.to(device)

    case_a.load_state_dict(
      {'weight': nn.Softmax(dim=0)(weight_total)}, strict=False)
    if t == 1:
      case_b.load_state_dict(
        {'weight': nn.Softmax(dim=0)(torch.Tensor([0., 0.]))}, strict=False)
    else:
      case_b.load_state_dict(
        {'weight': nn.Softmax(dim=0)(prev_weight)}, strict=False)
    
    logit_total, rad_total = case_a(val_inputs)
    logit_prev, rad_prev = case_a(val_inputs)
    _objective_total = criterion(
      trained_logits=logit_total,
      labels=val_labels, weight=case_a.weight,
      mode='eval', penalize=rad_total, device=device)
    _objective_prev = criterion(
      trained_logits=logit_prev, 
      labels=val_labels, weight=case_b.weight,
      mode='eval', penalize=rad_prev, device=device)
    val_i += 1
    val_total += _objective_total.item()
    val_prev += _objective_prev.item()
    
    del val_inputs, val_labels
    
  objective_prev = val_prev / val_i
  objective_total = val_total / val_i
  
  print("Objective_prev:", objective_prev, "Objective_total:", objective_total)
  
  if objective_prev >= objective_total:
    f_t = copy.copy(case_a)
    torch.save({
      'iter': t,
      'h_or_hprime': current_w,
      'model_state_dict': f_t.state_dict(),
      'min_objective': min_objective[current_w]},
      f=base_path + "/{}_checkpoint.pt".format(t))
  else:
    f_t = copy.copy(case_b)
    torch.save({
      'iter': t,
      'h_or_hprime': current_w,
      'model_state_dict': f_t.state_dict(),
      'min_objective': min_objective[current_w]},
      f=base_path + "/{}_checkpoint.pt".format(t))
    print("End iteration.")
    break