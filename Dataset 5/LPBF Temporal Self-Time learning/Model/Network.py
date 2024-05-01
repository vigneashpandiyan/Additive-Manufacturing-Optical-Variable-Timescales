# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch
The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"
@any reuse of this code should be authorized by the code author
"""

from torch.nn import Parameter
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
from torch import nn
sys.path.append("..")

# -*- coding: utf-8 -*-


sys.path.append("..")


class Linear_(nn.Module):
    def __init__(self, feature, nb_class,):
        super(Linear_, self).__init__()
        self.feature = feature
        # self.dropout = 0.1
        self.nb_class = nb_class

        self.fc1 = nn.Linear(self.feature, 32)
        self.fc2 = nn.Linear(32,  self.nb_class)

        # self.conv1 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        # )

    def forward(self, x):

        x = F.relu(self.fc1(x))
        # print("Xshape", x.shape)
        # x = self.conv1(x)
        # print("Xshape", x.shape)
        x = self.fc2(x)

        return x


# CausalCNNEncoder
# CausalCNN
# CausalConvolutionBlock


class PrintLayer(torch.nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        # print("Mainblock..", x.shape)
        return x


class PrintLayer_2(torch.nn.Module):
    def __init__(self):
        super(PrintLayer_2, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        # print("CausalConvolutionBlock_conv", x.shape)
        return x


class PrintLayer_3(torch.nn.Module):
    def __init__(self):
        super(PrintLayer_3, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        # print("CausalConvolutionBlock_chomp", x.shape)
        return x


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):

        # print("Chomp_size", self.chomp_size)
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """

    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """

    '''in_channels 1, 
       channels 10, 
       depth 1, 
       reduced_size 10,
       out_channels 10, 
       kernel_size 4'''

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # print(dilation)
        # print(kernel_size)
        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(conv1, chomp1, relu1, conv2, chomp2, relu2,
                                          )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        # print("CausalConvolutionBlock")
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """

    '''in_channels 1, 
       channels 10, 
       depth 1, 
       reduced_size 10,
       out_channels 10, 
       kernel_size 4'''

    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            # print("CausalCNN",i)
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        # print("CausalCNN")
        return self.network(x)


class TemporalCNN(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """

    '''in_channels 1, 
       channels 10, 
       depth 1, 
       reduced_size 10,
       out_channels 10, 
       kernel_size 4'''
    '''
    channels= 10
    depth= 5   
    in_channels= 1
    kernel_size= 17  
    out_channels= 3
    reduced_size= 80
    
    '''

    def __init__(self, reduced_size, in_channels=1, channels=8, depth=2,
                 kernel_size=17):
        super(TemporalCNN, self).__init__()

        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        # linear = torch.nn.Linear(reduced_size, out_channels)

        self.network = torch.nn.Sequential(
            causal_cnn, PrintLayer(), reduce_size, PrintLayer(), squeeze, PrintLayer(),
        )

    def get_params(self, deep=True):
        return {



            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,

        }

    def forward(self, x):
        # print("CausalCNNEncoder")
        x = x.view(x.shape[0], 1, -1)
        # print("xshapebefore the network", x.shape)
        return self.network(x)
