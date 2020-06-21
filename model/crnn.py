import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_class):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential()
        self.cnn.add_module('conv0', nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1))
        self.cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        self.cnn.add_module('conv1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.cnn.add_module('conv3', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.cnn.add_module('conv4', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.cnn.add_module('pooling2', nn.MaxPool2d(kernel_size=(1,2), stride=2))
        self.cnn.add_module('conv5', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        self.cnn.add_module('batchnorm0', nn.BatchNorm2d(512))
        self.cnn.add_module('conv6', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.cnn.add_module('batchnorm1', nn.BatchNorm2d(512))
        self.cnn.add_module('pooling3', nn.MaxPool2d(kernel_size=(1,2), stride=2))
        self.cnn.add_module('conv7', nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0))

        self.rnn = nn.Sequential()
        self.rnn.add_module("lstm0", nn.LSTM(512, 256, bidirectional=True))
        self.rnn.add_module("lstm1", nn.LSTM(512, 256, bidirectional=True))

        self.fc = nn.Linear(256*2, num_class+1)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        out = self.cnn(x)
        out = self.__map2seq(out)
        out = self.rnn(out)
        out = self.fc(out)
        out = self.softmax(out)

        return out


    def __map2seq(self, conv_out):
        N, C, H, W = conv_out.size()
        assert H == 1
        rnn_in = conv_out.squeeze(2)

        # W, N, C
        rnn_in = rnn_in.permute(2, 0, 1)

        return rnn_in
