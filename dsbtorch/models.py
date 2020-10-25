import numpy as np
import torch.nn as nn
import skimage

class SingleFrameClassifierModel(nn.Module):
    def __init__(self, img_channels=3):
        super(Model, self).__init__()
        # Layer 1: Conv + BatchNorm + Max Pool
        self.layer1 = self.conv_norm_pool(img_channels, 96, kernel_size=11, stride=3)
        # Layer 2: Conv + BatchNorm + Max Pool
        self.layer2 = self.conv_norm_pool(96, 256, kernel_size=5)
        # Layer 3: Three Convs + Max Pool
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3),
            nn.Conv2d(384, 384, kernel_size=3),
            nn.Conv2d(384, 256, kernel_size=3)
            nn.MaxPool2d(2, stride=2),
        )
        # Output layer: Flatten, FC's, Sigmoid
        in_features = 1  # What's the size of the flattened thing?
        self.out_layer = nn.Sequential(
            nn.Flatten(),  # Maintains minibatches
	        nn.linear(in_features, 4096),
	        nn.linear(4096, 4096),
	        nn.Sigmoid(),
        )
        

    def forward(self, x):
    	# Layer 1
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.out_layer(x)

    def conv_norm_pool(self, in_channels, out_chanels, **kwargs):
    	# Note: Torch expects dimensions to be (N, C, H, W) not (N, H, W, C)
    	block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_chanels),
            nn.MaxPool2d(2, stride=2),
        )

        return block