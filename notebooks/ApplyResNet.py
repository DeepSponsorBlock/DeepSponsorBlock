#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import sys
sys.path.append("../")


# In[2]:


from dsbtorch import IterableVideoSlidingWindowDataset, VideoSlidingWindowDataset, scan_dataset


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
import tqdm
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


# In[7]:


dataset_names = ['dev', 'test']  # ['train', 'dev', 'test']


# In[8]:


scanned_datasets = {}
for x in dataset_names:
    sdfile = pathlib.Path("scans/" + os.path.join(data_dir, x, "ws1-psr1-nsr1.pkl"))
    print(sdfile)
    if not sdfile.exists():
        raise ValueError("You need to use the ScanDatasets notebook first to scan & pickle the dataset.")
    with open(sdfile, 'rb') as f:
        scanned_datasets[x] = pickle.load(f)


# In[9]:


datasets = {x: VideoSlidingWindowDataset(scanned_datasets[x], t) for x in dataset_names}


# In[10]:


print("Training example count: %d" % len(datasets['dev']))


# In[11]:


output_dir = "/home/ubuntu/data/lstm_dataset/"


# In[12]:


model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load("/home/ubuntu/data/DeepSponsorBlock/results/attempt2-resnet50-sr10-sgd-lr10-decay1.weights"))


# In[25]:


# Remove the FC layer.
model.fc = nn.Identity()

for param in model.parameters():
    param.requires_grad = False


# In[26]:


model = model.to(device)
model.eval()


# In[27]:


batch_size = 1024
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, num_workers=6, pin_memory=True) for x in dataset_names}
dataset_sizes = {x: len(datasets[x]) for x in dataset_names}


# In[28]:


for x in dataset_names:
    sd = scanned_datasets[x]

    out_files = [output_dir / dir_list[0].parent.relative_to(data_dir).with_suffix(".pkl") for dir_list in sd.cumulative_dirs]
    lengths = list(np.diff(np.array(sd.cumulative_indices + [sd.n_indices])))
    
    # Reverse them to use as a stack.
    out_files.reverse()
    lengths.reverse()
    
    outputs = []
    labels = []
    for imgs, lbls in tqdm.tqdm(dataloaders[x]):
        imgs = torch.reshape(imgs, (-1, 3, 144, 256)).to(device)
        outputs.extend(list(model(imgs).cpu().numpy()))
        labels.extend(list(lbls.numpy()))
        
        while len(outputs) >= lengths[-1]:
            out_path = out_files.pop()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'wb') as f:
                length = lengths.pop()
                out = (outputs[:length], labels[:length])
                pickle.dump(out, f)
                
                outputs = outputs[length:]
                labels = labels[length:]
            # print("Wrote %s" % str(out_path))


# In[ ]:




