from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import pdb

from datasets.mnist import MNIST
from models.mnist_net import Net

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def train(cfgs, model, device, data_loader, optimizer, cur_epoch, writer):
    model.train()
    criterion = F.cross_entropy
    for batch_idx, (data, target, _) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # writer.add_scalar('Loss/train',loss.data, batch_idx)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    return model

def test(args, model, device, data_loader, writer, episode_id, is_val=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    if is_val:
        for data, target, _ in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
    else:
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

    test_loss /= total
    test_acc = 100. * correct / total 

    # labeled set size
    ls_size = args.init_size + (episode_id * args.al_batch_size)
    
    # writer.add_scalar('Loss/test', test_loss, ls_size)
    # writer.add_scalar('Accuracy/test', test_acc , ls_size)
    set_name = ''
    if is_val:
        set_name = 'Validation Set'
    else:
        set_name = 'Test set'
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set_name, test_loss, correct, total,
        test_acc))

    return test_acc
