import torch
import torch.nn as nn
import torchvision.models as models


def ResCNNClassifier(weights_path=None, **kwargs):
    """A two-class classifier based on the ResNet50 model, with
    pretrained weights loaded from the file (or ImageNet ResNet50 weights used
    if no weights file is provided)."""

    resnet = models.resnet50(pretrained=(not weights_path))
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)

    if weights_path:
        resnet.load_state_dict(torch.load(weights_path, **kwargs))

    return resnet


def ResCNN(weights_path=None, **kwargs):
    """A ResNet50-based model that outputs the result of the last convolutional
    layer. Can be used alone to pre-encode images, or can be used in a
    Sequential module to construct an end-to-end model."""

    resnet = ResCNNClassifier(weights_path, **kwargs)

    out_features = resnet.fc.in_features
    resnet.fc = nn.Identity()

    return resnet


class Encoder(nn.Module):
    """A sequence of Linear and BatchNorm modules to convert the frozen ResNet50
    convolution output to a learnable vector encoding for the frame. Supports
    with PackedSequence objects."""

    def __init__(self, in_features, fc_hidden1=512, fc_hidden2=512,
                 CNN_embed_dim=300):
        super(Encoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2

        self.fc = nn.Sequential(
            nn.Linear(in_features, fc_hidden1),
            nn.BatchNorm1d(fc_hidden1, momentum=0.01),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.BatchNorm1d(fc_hidden2, momentum=0.01),
            nn.Linear(fc_hidden2, CNN_embed_dim),
        )

    def forward(self, x):
        is_packed = isinstance(x, nn.utils.rnn.PackedSequence)
        if is_packed:
            x, batch_sizes, sorted_indices, unsorted_indices = x

        out = self.fc(x)

        if is_packed:
            out = nn.utils.rnn.PackedSequence(out, batch_sizes, sorted_indices,
                                              unsorted_indices)
        else:
            out = torch.unsqueeze(out, 0)

        return out


def ResCNNEncoder(fc_hidden1=512, fc_hidden2=512, CNN_embed_dim=300,
                  weights_path=None, **kwargs):
    """A full frame-to-vector encoder produced by combining the ResCNN with
    the Encoder sequentially."""

    resnet = ResCNN(weights_path, **kwargs)
    encoder = Encoder(resnet.fc.in_features, fc_hidden1, fc_hidden2,
                        CNN_embed_dim)
    return nn.Sequential(
        resnet, encoder
    )


class DecoderRNN(nn.Module):
    """A Bidirectional LSTM-based decoder for producing sponsored segment
    start and end labels using input feature vectors of all frames in video."""

    def __init__(self, CNN_embed_dim=300, h_RNN_layers=2, h_RNN=256,
                 sigmoid=True):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes

        self.start_LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,
            bidirectional=True
        )

        self.end_LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc_start = nn.Linear(2 * self.h_RNN, 1)
        self.fc_end = nn.Linear(2 * self.h_RNN, 1)
        self.sigmoid = nn.Sigmoid() if sigmoid else None

    def forward_LSTM(self, LSTM, fc, x_RNN):
        LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = LSTM(x_RNN, None)

        is_packed = isinstance(RNN_out, nn.utils.rnn.PackedSequence)
        if is_packed:
            RNN_out, batch_sizes, sorted_indices, unsorted_indices = RNN_out

        x = fc(RNN_out)
        if self.sigmoid:
            x = self.sigmoid(x)

        if is_packed:
            x = nn.utils.rnn.PackedSequence(x, batch_sizes, sorted_indices,
                                            unsorted_indices)

        return x

    def forward(self, x_RNN):
        start_x = self.forward_LSTM(self.start_LSTM, self.fc_start, x_RNN)
        end_x = self.forward_LSTM(self.end_LSTM, self.fc_end, x_RNN)
        return start_x, end_x


def PreprocessedEncoderDecoder(
        in_features, fc_hidden1=512, fc_hidden2=512, CNN_embed_dim=300,
        h_RNN_layers=2, h_RNN=256, sigmoid=True,
        weights_path=None, **kwargs):
    """A module combining the encoder (converting ResCNN output to encoding) and
    the decoder (converting the encoding to the start/end labels). For use with
    preprocessed data, e.g. input data should be the output of the ResCNN on
    the input frames."""

    encoder = Encoder(
        in_features, fc_hidden1, fc_hidden2, CNN_embed_dim)
    decoder = DecoderRNN(CNN_embed_dim, h_RNN_layers, h_RNN, sigmoid)

    model = nn.Sequential(encoder, decoder)

    if weights_path:
        model.load_state_dict(torch.load(weights_path, **kwargs))

    return model
