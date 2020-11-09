#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import sys
sys.path.append("../")


# In[2]:


from dsbtorch import IterableVideoSlidingWindowDataset, VideoSlidingWindowDataset


# In[3]:


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
import pathlib
import pickle
from torch.utils.tensorboard import SummaryWriter

plt.ion()   # interactive mode


# In[4]:


device = torch.device("cuda")
torchvision.set_image_backend('accimage')


# In[5]:


t = transforms.Compose([
    transforms.Resize((144, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[6]:


data_dir = "/home/ubuntu/data/dataset/"
# data_dir = "/home/ubuntu/data/toy_dataset/"
# data_dir = "/home/ubuntu/data/dataset_imagefolder_sampled_1000/"
# data_dir = "/home/ubuntu/data/toy_dataset_imagefolder/"
is_imagefolder = False
sampling_rate = 10


# In[7]:


dataset_names = ['train', 'dev'] #, 'test']

# In[8]:


window_size = 1
positive_sampling_rate = sampling_rate
negative_sampling_rate = 10 * sampling_rate


# In[9]:


if not is_imagefolder:
    scanned_datasets = {}
    for x in dataset_names:
        sddir = pathlib.Path("scans/" + os.path.join(data_dir, x))
        sdfile = sddir / ("ws%d-psr%d-nsr%d.pkl" % (window_size, positive_sampling_rate, negative_sampling_rate))
        print(sdfile)
        if not sdfile.exists():
            raise ValueError("You need to use the ScanDatasets notebook first to scan & pickle the dataset.")
        with open(sdfile, 'rb') as f:
            scanned_datasets[x] = pickle.load(f)


# In[10]:


if is_imagefolder:
    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), transform=t) for x in dataset_names}
else:
    datasets = {x: VideoSlidingWindowDataset(scanned_datasets[x], t) for x in dataset_names}


# In[11]:


print("Training example count: %d" % len(datasets['train']))


# In[12]:


batch_size = 128
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, num_workers=6, pin_memory=False) for x in dataset_names}
dataset_sizes = {x: len(datasets[x]) for x in dataset_names}


# In[13]:


def running_totals(preds, labels, tp, tn, fp, fn):
    tp += torch.sum((preds == labels) * (labels == 1)).item()
    tn += torch.sum((preds == labels) * (labels == 0)).item()
    fp += torch.sum((preds != labels) * (labels == 0)).item()
    fn += torch.sum((preds != labels) * (labels == 1)).item()
    return (tp, tn, fp, fn)


# In[14]:


def train_model(model, criterion, optimizer, scheduler, output_path, num_epochs=25, beta2=0.25, print_every_n=0, writer=None, hyperparams=None):
    writer = SummaryWriter("runs/resnet50/%s-lr%d-decay%d" % (hyperparams['optimizer'], int(hyperparams['learning_rate'] * 1000), int(hyperparams['lr_decay'])))
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_fscore = 0.0
    
    epoch_loss = {}
    epoch_fscore = {}
    epoch_accuracy = {}

    for epoch in range(num_epochs):
        print('\n\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print('\nTraining phase.')
                print('-' * 8)
            else:
                model.eval()   # Set model to evaluate mode
                print('\nEvaluation phase.')
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
                
                if (print_every_n > 0 and i % print_every_n == 0 and i > 0) or i == len(dataloaders[phase]) - 1:
                    td = time.time() - batch_start
                    print("Batch number ", i)
                    print("Total images used this epoch: ", batch_size * print_every_n)
                    print("Statistics: ", tp, tn, fp, fn)
                    print("Time since last update: ", td)
                    print("Time per 1000 images:", td * 1000 / (batch_size * print_every_n))
                    batch_start = time.time()

                i += 1
                
            if phase == 'train':
                scheduler.step()

            epoch_loss[phase] = running_loss / dataset_sizes[phase]
            epoch_fscore[phase] = (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp)
            epoch_accuracy[phase] = (tp + tn) / (tp + tn + fp + fn)
            
            writer.add_scalar("Loss/" + phase, epoch_loss[phase], epoch)
            writer.add_scalar("FScore/" + phase, epoch_fscore[phase], epoch)
            writer.add_scalar("Accuracy/" + phase, epoch_accuracy[phase], epoch)

            print('{} Loss: {:.4f} F0.5: {:.4f}  Acc: {:.4f}'.format(
                phase, epoch_loss[phase], epoch_fscore[phase], epoch_accuracy[phase]))

            # deep copy the model
            if phase == 'dev' and epoch_fscore[phase] > best_fscore:
                best_fscore = epoch_fscore[phase]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, output_path + str(epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F score: {:4f}'.format(best_fscore))
    
    if hyperparams:
        writer.add_hparams(
            hyperparams,
            {
                "H-Loss/Train": epoch_loss['train'],
                "H-Loss/Dev": epoch_loss['dev'],
                "H-FScore/Train": epoch_fscore['train'],
                "H-FScore/Dev": epoch_fscore['dev'],
                "H-Accuracy/Train": epoch_accuracy['train'],
                "H-Accuracy/Dev": epoch_accuracy['dev'],
            })
        
    writer.close()

    # Save and load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), output_path)
    return model


# In[ ]:


num_epochs = 20

for optname in ['sgd']: # ['adam', 'sgd']:
    for lr in [0.01]: # [0.1, 0.01, 0.001]:
        for decay in [True]: # [True, False]:
            model = models.resnet50(pretrained=True)
            # for param in model.parameters():
            #     param.requires_grad = False

            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)

            model = model.to(device)

            criterion = nn.CrossEntropyLoss()

            # Observe that all parameters are being optimized
            if optname == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=(0.1 if decay else 1))
            
            hyperparams = {'optimizer': optname, 'learning_rate': lr, 'lr_decay': decay}

            outpath = "../results/resnet50-sr%d-%s-lr%d-decay%d.weights" % (sampling_rate, optname, int(lr * 1000), int(decay))
            model = train_model(model, criterion, optimizer, exp_lr_scheduler, outpath, num_epochs=num_epochs, print_every_n=50, hyperparams=hyperparams)
