#!/usr/bin/env python
# coding: utf-8

# In[32]:


from __future__ import print_function, division
import sys
sys.path.append("../")


# In[33]:


from dsbtorch import scan_dataset, IterableVideoSlidingWindowDataset, VideoSlidingWindowDataset


# In[34]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode


# In[35]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[36]:


t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[37]:


data_dir = "/home/ubuntu/data/dataset/"
# data_dir = "/home/ubuntu/data/toy_dataset/"
dataset_names = ['train', 'dev', 'test']
scanned_datasets = {x: scan_dataset(os.path.join(data_dir, x), 1, 1, 10) for x in dataset_names}
datasets = {x: VideoSlidingWindowDataset(scanned_datasets[x], t) for x in dataset_names}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=128, num_workers=6) for x in dataset_names}
dataset_sizes = {x: len(datasets[x]) for x in dataset_names}


# In[38]:


def running_totals(preds, labels, tp, tn, fp, fn):
    tp += torch.sum((preds == labels) * (labels == 1)).item()
    tn += torch.sum((preds == labels) * (labels == 0)).item()
    fp += torch.sum((preds != labels) * (labels == 0)).item()
    fn += torch.sum((preds != labels) * (labels == 1)).item()
    return (tp, tn, fp, fn)


# In[39]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, beta2=0.25, print_every_n=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_fscore = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print('Training for one epoch.')
                print('-' * 8)
            else:
                model.eval()   # Set model to evaluate mode
                print('Evaluating model.')
                print('-' * 8)

            running_loss = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0

            i = 0
            batch_start = time.time()
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:                        
                inputs = torch.reshape(inputs, (-1, 3, 144, 256))
                inputs = inputs.to(device)
                labels = torch.reshape(labels, (-1, ))
                labels = labels.long()
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                tp, tn, fp, fn = running_totals(preds, labels.data, tp, tn, fp, fn)
                
                if i % print_every_n == 0 and i > 0:
                    print("Batch number ", i)
                    print("Statistics: ", tp, tn, fp, fn)
                    print("Time since last update: ", time.time() - batch_start)
                    batch_start = time.time()
                i += 1
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_fscore = (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp)

            print('{} Loss: {:.4f} F0.5: {:.4f}'.format(
                phase, epoch_loss, epoch_fscore))

            # deep copy the model
            if phase == 'val' and epoch_fscore > best_fscore:
                best_fscore = epoch_fscore
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F score: {:4f}'.format(best_fscore))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[40]:


model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# In[41]:


model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=1, print_every_n=10)


# In[ ]:




