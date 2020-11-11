#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import sys
sys.path.append("../")

from dsbtorch import scan_dataset, IterableVideoSlidingWindowDataset, VideoSlidingWindowDataset

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

import tqdm

from CNN_RNN import *
from torch.utils.tensorboard import SummaryWriter


# In[2]:


plt.ion()   # interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

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

window_size = 1
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


# In[4]:


batch_size = 1024
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, num_workers=6, pin_memory=True) for x in dataset_names}
dataset_sizes = {x: len(datasets[x]) for x in dataset_names}


# In[5]:


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


# In[6]:


def train_model(model, criterion, optimizer, scheduler, output_path, num_epochs=25, beta2=0.25, print_every_n=0):
    writer = SummaryWriter()

    since = time.time()
    cnn_encoder, rnn_decoder = model

    best_decoder_wts = copy.deepcopy(rnn_decoder.state_dict())
    best_iou = 0.0

    for epoch in range(num_epochs):
        print('\n\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        epoch_loss = {}
        epoch_iou = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            cnn_encoder.eval()
            if phase == 'train':
                rnn_decoder.train()
                print('Training for one epoch.')
                print('-' * 8)
            else:
                rnn_decoder.eval()
                print('Evaluating model.')
                print('-' * 8)

            running_loss = 0.0
            total_iou = 0

            i = 0
            video_start = time.time()
            
            sd = scanned_datasets[phase]

            lengths = list(np.diff(np.array(sd.cumulative_indices + [sd.n_indices])))

            # Reverse them to use as a stack.
            lengths.reverse()

            encoder_outputs = []
            acc_labels = []
            for imgs, lbls in tqdm.tqdm(dataloaders[phase]):
                imgs = torch.reshape(imgs, (-1, 3, 144, 256)).to(device)
                encoder_outputs.append(encoder(imgs))
                acc_labels.append(torch.reshape(lbls, (-1, )).to(device))

                while lengths and sum(x.shape[0] for x in encoder_outputs) >= lengths[-1]:
                    combined_encoder_outputs = torch.cat(encoder_outputs).to(device)
                    combined_labels = torch.cat(acc_labels).to(device)
                    
                    length = lengths.pop()
                    
                    encoder_outputs = [combined_encoder_outputs[length:]]
                    acc_labels = [combined_labels[length:]]
                    
                    cnn_outputs, labels = (combined_encoder_outputs[:length], combined_labels[:length])
                    
                    # labels = torch.reshape(labels, (-1, ))
                    diffs = (labels[1:] - labels[:-1]).to(device)

                    # 1's for frames where the start of a sponsored segment occurs
                    start_labels = torch.cat((torch.tensor([labels[0]]).to(device), diffs)) == 1
                    start_labels = start_labels.float()
                    start_labels = start_labels.to(device)

                    # 1's for frames where the end of a sponsored segment occurs
                    end_labels = torch.cat((diffs, torch.tensor([labels[-1]]).to(device))) == -1
                    end_labels = end_labels.float()                
                    end_labels = end_labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        start_probs, end_probs = rnn_decoder(cnn_outputs)
                        start_probs = torch.reshape(start_probs, (-1,)).to(device)
                        end_probs = torch.reshape(end_probs, (-1,)).to(device)

                        start_preds = torch.gt(start_probs, 0.5).to(device)
                        end_preds = torch.gt(end_probs, 0.5).to(device)

                        loss = criterion(start_probs, start_labels) + criterion(end_probs, end_labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    preds = convert_to_onehot(start_preds, end_preds).to(device)
                    total_iou += IOU(preds, labels)

                    if print_every_n > 0 and i % print_every_n == 0 and i > 0:
                        print("Video number ", i + 1)
                        print("Time since last update: ", time.time() - video_start)
                        video_start = time.time()
                    i += 1
                    
            assert not lengths
            assert sum(x.shape[0] for x in encoder_outputs) == 0
            assert sum(x.shape[0] for x in acc_labels) == 0
                
            if phase == 'train':
                scheduler.step()

            epoch_loss[phase] = running_loss / i
            epoch_iou[phase] = total_iou / i
            
            writer.add_scalar("Loss/" + phase, epoch_loss[phase], epoch)
            writer.add_scalar("IOU/" + phase, epoch_iou[phase], epoch)

            print('{} Loss: {:.4f} IOU: {:.4f}'.format(
                phase, epoch_loss[phase], epoch_iou[phase]))

            # deep copy the model
            if phase == 'dev' and epoch_iou[phase] > best_iou:
                best_iou = epoch_iou[phase]
                best_decoder_wts = copy.deepcopy(rnn_decoder.state_dict())
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F score: {:4f}'.format(best_iou))

    # Save and load best model weights
    rnn_decoder.load_state_dict(best_decoder_wts)
    torch.save(rnn_decoder.state_dict(), output_path + ".decoder")
    return (cnn_encoder, rnn_decoder)


# In[7]:


encoder = ResCNN("/home/ubuntu/data/DeepSponsorBlock/results/attempt2-resnet50-sr10-sgd-lr10-decay1.weights")
decoder = nn.Sequential(
    Embedder(encoder.num_ftrs),
    DecoderRNN()
)


# In[ ]:


encoder = encoder.to(device)
decoder = decoder.to(device)

criterion = nn.BCELoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(decoder.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model((encoder, decoder), criterion, optimizer, exp_lr_scheduler, '../results/rnn.weights', num_epochs=20, print_every_n=0)

