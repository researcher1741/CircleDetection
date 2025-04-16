"""
This module defines the architecture of the CircleRegression model.

It consists of a Convolutional Neural Network (CNN) backbone followed by a fully connected feed-forward network (FFN).
The CNN extracts spatial features from the noisy circle image, and the FFN regresses the circle parameters (row, col, radius).
The model supports configurable convolutional layers and dimensions, and uses GELU activation throughout.
"""

# DL modules
import torch.nn as nn


class CircleRegression(nn.Module):
    """
    CircleRegression predicts (row, col, radius) of a noisy circle in a grayscale image.
    It combines a CNN backbone with a final FFN for regression.
    Inputs:
        input_shape (tuple): Shape of the input image, e.g., (1, 100, 100)
        channels (tuple): Number of channels at each CNN layer
        kernels (tuple): Kernel size for each CNN layer
        pools (tuple): Pool size for each CNN layer
        strides (tuple or None): Strides for each pool layer
        FFN_dims (tuple): Dimensions of the fully connected layers including output layer (e.g., (1024, 128, 3))
    """

    def __init__(self, input_shape=(1, 100, 100),
                 channels=(1, 128, 512, 64),
                 kernels=(5, 3, 3),
                 pools=(2, 2, 2),
                 strides=None,
                 FFN_dims=(1024, 128, 3),
                 ):
        super(CircleRegression, self).__init__()
        if not strides:
            strides = pools

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pools = nn.ModuleList()

        # CNN layers
        for i in range(len(kernels)):
            self.convs.append(nn.Conv2d(channels[i], channels[i + 1], kernels[i]))
            self.bns.append(nn.BatchNorm2d(channels[i + 1]))
            self.pools.append(nn.MaxPool2d(pools[i], strides[i]))

        # Compute output size from CNN
        cnn_out_dim = self.get_cv_output_dim(input_shape[1], kernels, pools, strides)
        fc_input_size = channels[-1] * cnn_out_dim * cnn_out_dim

        # FFN layers
        self.fcs = nn.ModuleList()
        for in_dim, out_dim in zip((fc_input_size, *FFN_dims[:-1]), FFN_dims):
            self.fcs.append(nn.Linear(in_dim, out_dim))

        self.act_cv = nn.GELU()

        # Weight initialization for GELU
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Custom weight initialization for GELU.
        Kaiming initialization is more appropriate than Xavier for GELU activations (default one).
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def get_cv_output_dim(self, H, kernels, pools, strides, stride=1, padding=0, dilation=1):
        """
        Computes the spatial resolution after all convolution and pooling layers.
        """
        for k, p, s in zip(kernels, pools, strides):
            H = self.compute_output_size_conv2d(H, k, stride, padding, dilation)
            H = self.compute_output_size_pool(H, p, s)
        return H

    @staticmethod
    def compute_output_size_conv2d(H_in, kernel_size, stride=1, padding=0, dilation=1):
        """ Computes the output size after a convolution layer. """
        return (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def compute_output_size_pool(H_in, kernel_size, stride=None, padding=0):
        """ Computes the output size after a pooling layer. """
        if stride is None:
            stride = kernel_size
        return (H_in + 2 * padding - kernel_size) // stride + 1

    def forward(self, x):
        """ Forward pass through the model (CNN + FFN) """
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = pool(self.act_cv(bn(conv(x))))

        x = x.view(x.size(0), -1)  # Flatten before FC layers
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                x = self.act_cv(x)
        return x
