import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_network import BaseNetwork


class ConvBlock(nn.Module):
    """
    Down blocks in U-Net
    """

    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannels),
            nn.Conv2d(outchannels, outchannels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannels)
        )
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv_block(x)


class UpBlock(nn.Module):
    """ Up blocks in U-Net. Similar to the down blocks, but incorporates input from skip connections. """

    def __init__(self, inchannels, outchannels, kernel_size=2, stride=2):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            inchannels, outchannels, kernel_size=kernel_size, stride=stride
        )
        self.conv = ConvBlock(inchannels, outchannels)

    def forward(self, x, skips):
        x = self.upconv(x)
        x = torch.cat([skips, x], 1)
        return self.conv(x)


class UnetModel(BaseNetwork):
    def __init__(self, opts):
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.in_channels = opts.input_channels
        self.out_channels = opts.first_layer_filters
        self.net_depth = opts.net_depth
        self.num_classes = opts.num_classes

        # down transformations
        for _ in range(self.net_depth):
            conv = ConvBlock(self.in_channels, self.out_channels)
            self.downblocks.append(conv)
            self.in_channels, self.out_channels = self.out_channels, 2 * self.out_channels

        # midpoint
        self.middle_conv = ConvBlock(self.in_channels, self.out_channels)

        # up transformations
        self.in_channels, self.out_channels = self.out_channels, int(self.out_channels / 2)
        for _ in range(self.net_depth):
            upconv = UpBlock(self.in_channels, self.out_channels)
            self.upblocks.append(upconv)
            self.in_channels, self.out_channels = self.out_channels, int(self.out_channels / 2)

        self.seg_layer = nn.Conv2d(2 * self.out_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        decoder_outputs = []

        for op in self.downblocks:
            decoder_outputs.append(op(x))
            x = self.pool(decoder_outputs[-1])

        x = self.middle_conv(x)

        for op in self.upblocks:
            x = op(x, decoder_outputs.pop())
        return self.seg_layer(x)
    
    def forward_features(self, x):
        decoder_outputs = []

        for op in self.downblocks:
            decoder_outputs.append(op(x))
            x = self.pool(decoder_outputs[-1])

        x = self.middle_conv(x)

        for op in self.upblocks:
            x = op(x, decoder_outputs.pop())
        return self.seg_layer(x), x