import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# 2D CNN encoder using ResNet-152 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.resnet = models.resnet50(pretrained=True)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, fc_hidden1),
            nn.BatchNorm1d(fc_hidden1, momentum=0.01),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.BatchNorm1d(fc_hidden2, momentum=0.01),
            nn.Linear(fc_hidden2, CNN_embed_dim),
        )
        
    def forward(self, x):
        return torch.unsqueeze(self.resnet(x), 0)


class ResCNN(nn.Module):
    def __init__(self, weights_path):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNN, self).__init__()

        self.resnet = models.resnet50()
        self.num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.num_ftrs, 2)
        self.resnet.load_state_dict(torch.load(weights_path))

        self.resnet.fc = nn.Identity()

        for param in self.resnet.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        return self.resnet(x)


class Embedder(nn.Module):
    def __init__(self, in_features, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(Embedder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, fc_hidden1),
            nn.BatchNorm1d(fc_hidden1, momentum=0.01),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.BatchNorm1d(fc_hidden2, momentum=0.01),
            nn.Linear(fc_hidden2, CNN_embed_dim),
        )
        
    def forward(self, x):
        return torch.unsqueeze(self.fc(x), 0)


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=2, h_RNN=256, h_FC_dim=128, drop_p=0.3):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p

        self.start_LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True
        )

        self.end_LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True
        )

        self.fc_start = nn.Linear(2 * self.h_RNN, 1)
        self.fc_end = nn.Linear(2 * self.h_RNN, 1)

    def forward_LSTM(self, LSTM, fc, x_RNN):
        LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = fc(RNN_out)
        sig = nn.Sigmoid()
        x = sig(x)

        return x

    def forward(self, x_RNN):        
        start_x = self.forward_LSTM(self.start_LSTM, self.fc_start, x_RNN)
        end_x = self.forward_LSTM(self.end_LSTM, self.fc_end, x_RNN)
        return start_x, end_x