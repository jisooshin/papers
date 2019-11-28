#-*- coding:utf-8 -*-
import os
import sys
import argparse
from datetime import datetime

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models import resnet34

parser = argparse.ArgumentParser(description='Train ResNet34 model.')
parser.add_argument(
  '--base_path', type=str, default=os.environ['HOME'] + '/cifar100',
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
  '--patience', type=int, default=1000)
parser.add_argument(
  '--verbose', type=int, default=1)
parser.add_argument(
  '--name', type=str, default=datetime.strftime(datetime.now(), '%Y%m%d_%H'))
# Check parameter or train
parser.add_argument(
  '--check', type=str, default='TRUE')
ARGS = parser.parse_args()

# Check or make base path 
if os.path.isdir(ARGS.base_path):
  pass
else:
  os.makedirs(ARGS.base_path)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
trainlist = torch.utils.data.random_split(train, [40000, 10000])
train, val = trainlist[0], trainlist[1]
trainloader = torch.utils.data.DataLoader(
  train, batch_size=ARGS.train_batch, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(
  val, batch_size=ARGS.val_batch, shuffle=True, num_workers=0)

net = resnet34.ResNet34()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters())

if ARGS.check.upper() == 'TRUE':
  params = net.parameters()
  total_param = 0
  while True:
    try:
      total_param += next(params).shape.numel()
    except StopIteration:
      break
  print("# trainable parameters:{}".format(total_param))
  # NOTE:
  #print("# of running features(powered by batch size):{}".format(10)) 
else:
  writer = SummaryWriter(log_dir=ARGS.base_path + '/logs/{}'.format(ARGS.name))
  
  # Early Stopping
  max_patience = ARGS.patience
  patience = 0
  global_steps = 0

  for epoch in range(ARGS.max_epoch):
    running_loss = 0.
    early_metrics = []
    min_value = 0.
    for i, data in enumerate(trainloader):
      global_steps += 1

      if patience >= max_patience:
        print("Early Stopping.")
        break
      else:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = net(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        # running_loss += loss.item()
        
        if i % ARGS.verbose == (ARGS.verbose - 1):
          val_inputs, val_labels = next(iter(valloader))

          # Memory allocate to validate
          val_inputs, val_labels = val_inputs.to('cpu'), val_labels.to('cpu')
          val_net = resnet34.ResNet34()
          val_net = val_net.to('cpu')
          val_net.load_state_dict(net.state_dict())
          val_logits = val_net(val_inputs)
          val_loss = criterion(val_logits, val_labels)
          val_loss = val_loss.item() / ARGS.val_batch
          train_correct_case = (logits.max(dim=1)[1] == labels).sum()
          val_correct_case = (val_logits.max(dim=1)[1] == val_labels).sum()
          # print("train", logits.max(dim=1)[1], labels, "\nval", val_logits.max(dim=1)[1], val_labels)

          all_case = val_labels.shape[0]
          train_accuracy = (train_correct_case.item()/all_case)
          val_accuracy = (val_correct_case.item()/all_case)
          train_loss = loss.item() / ARGS.train_batch

          # NOTE: To keep memory usage
          del inputs, labels, val_inputs, val_labels

          # Early stop: have to select metric to apply early stopping method.
          _metric = train_loss
          if len(early_metrics) < 2:
            early_metrics.append(_metric) # Append metric 
          elif len(early_metrics) >= 2:
            if _metric > min_value:
              patience += 1
              print("Patience added: {}".format(patience))
            early_metrics.append(_metric)
            min_value = min(early_metrics)
            early_metrics.sort() # Sort ascending
            early_metrics = early_metrics[:2]
          else:
            print("Error occured.")
            break

          print(
            'Epoch[{}] Step[{}]:'.format(epoch + 1, global_steps), '\n\t',
            '[train loss]: {0:.3f}'.format(train_loss), '\n\t',
            '[train accuracy]: {0:.3f}'.format(train_accuracy), '\n\t',
            '[val loss]: {0:.3f}'.format(val_loss), '\n\t',
            '[val accuracy]: {0:.3f}'.format(val_accuracy))
          writer.add_scalar(
            tag='train loss', 
            scalar_value=train_loss, global_step=global_steps)
          writer.add_scalar(
            tag='train accuracy', 
            scalar_value=train_accuracy, global_step=global_steps)
          writer.add_scalar(
            tag='val loss',
            scalar_value=val_loss, global_step=global_steps)
          writer.add_scalar(
            tag='val accuracy',
            scalar_value=val_accuracy, global_step=global_steps)
           
          running_loss = 0.
    torch.save({
      'epoch': epoch,
      'model_state_dict': net.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss}, ARGS.base_path + "/{}_{}.pt".format(epoch, ARGS.name))    
  print("Finish.")