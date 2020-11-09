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
import pathlib
import pickle

from CNN_RNN import *
# from torch.utils.tensorboard import SummaryWriter

plt.ion()   # interactive mode

max_frames = 1200  # 20 min
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# data_dir = "/home/ubuntu/data/dataset/"
data_dir = "/home/ubuntu/data/toy_dataset/"
# data_dir = "/home/ubuntu/data/dataset_imagefolder_sampled_1000/"
# data_dir = "/home/ubuntu/data/toy_dataset_imagefolder/"
is_imagefolder = False
sampling_rate = 1

dataset_names = ['train', 'dev'] #, 'test']

window_size = 10
positive_sampling_rate = sampling_rate
negative_sampling_rate = sampling_rate

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
            
if is_imagefolder:
    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), transform=t) for x in dataset_names}
else:
    datasets = {x: VideoSlidingWindowDataset(scanned_datasets[x], t) for x in dataset_names}
    
batch_size = 8
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], num_workers=6, pin_memory=True) for x in dataset_names}
dataset_sizes = {x: len(datasets[x]) for x in dataset_names}


def convert_to_onehot(start_preds, end_preds):
    """
    Converts preds into a one-hot vector with 1's for sponsored frames
    """
    onehot_preds = np.zeros(len(start_preds))

    # Transform into tuples of (timestamp, is_start)
    start_preds = [(idx, True) for idx in torch.nonzero(start_preds, as_tuple=False)]
    end_preds = [(idx, False) for idx in torch.nonzero(end_preds, as_tuple=False)]
    timestamps = sorted(start_preds + end_preds, key=lambda t: t[0])

    seg_start = 0
    in_seg = False
    for t, is_start in timestamps:
        if is_start and not in_seg:
            seg_start = t
            in_seg = True
        elif not is_start and in_seg:
            onehot_preds[seg_start : t+1] = 1
            in_seg = False

    return torch.tensor(onehot_preds, dtype=torch.long)


def IOU(preds, labels):
    intersection = torch.nonzero(preds * labels, as_tuple=False).shape[0]
    union = torch.nonzero(preds + labels, as_tuple=False).shape[0]
    return intersection / union if union != 0 else 0


def train_model(model, criterion, optimizer, scheduler, output_path, num_epochs=25, beta2=0.25, print_every_n=0, max_frames=max_frames):
    since = time.time()
    cnn_encoder, rnn_decoder = model

    best_encoder_wts = copy.deepcopy(cnn_encoder.state_dict())
    best_decoder_wts = copy.deepcopy(rnn_decoder.state_dict())
    best_iou = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                cnn_encoder.train()  # Set model to training mode
                rnn_decoder.train()
                print('Training for one epoch.')
                print('-' * 8)
            else:
                cnn_encoder.eval()   # Set model to evaluate mode
                rnn_decoder.eval()
                print('Evaluating model.')
                print('-' * 8)

            running_loss = 0.0
            total_iou = 0

            i = 0
            batch_start = time.time()
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:               
                inputs = torch.reshape(inputs, (-1, 3, 144, 256))
                # inputs = inputs[:max_frames]
                inputs = inputs.to(device)

                labels = torch.reshape(labels, (-1, ))
                # labels = labels[:max_frames]
                diffs = labels[1:] - labels[:-1]
                
                # 1's for frames where the start of a sponsored segment occurs
                start_labels = torch.cat((torch.tensor([labels[0]]), diffs)) == 1
                start_labels = start_labels.float()
                start_labels = start_labels.to(device)
                
                # 1's for frames where the end of a sponsored segment occurs
                end_labels = torch.cat((diffs, torch.tensor([labels[-1]]))) == -1
                end_labels = end_labels.float()                
                end_labels = end_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    cnn_outputs = cnn_encoder(inputs)
                    
                    start_probs, end_probs = rnn_decoder(cnn_outputs)
                    start_probs = torch.reshape(start_probs, (-1,))
                    end_probs = torch.reshape(end_probs, (-1,))
                    
                    start_preds = torch.gt(start_probs, 0.5)
                    end_preds = torch.gt(end_probs, 0.5)
                    
                    loss = criterion(start_probs, start_labels) + criterion(end_probs, end_labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                preds = convert_to_onehot(start_preds, end_preds)
                total_iou += IOU(preds, labels)
                
                if print_every_n > 0 and i % print_every_n == 0 and i > 0:
                    print("Batch number ", i)
                    print("Time since last update: ", time.time() - batch_start)
                    batch_start = time.time()
                i += 1
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_iou = total_iou / dataset_sizes[phase]

            print('{} Loss: {:.4f} F0.5: {:.4f}'.format(
                phase, epoch_loss, epoch_iou))

            # deep copy the model
            if phase == 'dev' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_encoder_wts = copy.deepcopy(cnn_encoder.state_dict())
                best_decoder_wts = copy.deepcopy(rnn_decoder.state_dict())
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F score: {:4f}'.format(best_iou))

    # Save and load best model weights
    cnn_encoder.load_state_dict(best_encoder_wts)
    rnn_decoder.load_state_dict(best_decoder_wts)
    torch.save(cnn_encoder.state_dict(), output_path + ".encoder")
    torch.save(rnn_decoder.state_dict(), output_path + ".decoder")
    return (cnn_encoder, rnn_decoder)



encoder = ResCNNEncoder()
decoder = DecoderRNN(max_frames=max_frames)

encoder = encoder.to(device)
decoder = decoder.to(device)

criterion = nn.BCELoss()

# Observe that all parameters are being optimized
params = list(decoder.parameters()) + list(encoder.parameters())
optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model((encoder, decoder), criterion, optimizer, exp_lr_scheduler, '../results/rnn.weights', num_epochs=1, print_every_n=10)

