#!/usr/bin/env python
import torch
import torch.nn as nn
import json
import os
import pytorch.utils.pytorch_model_utils as nn_utils

class Down(nn.Module):
    """
    Down blocks in U-Net
    """
    def __init__(self, conv, max):
        super(Down, self).__init__()
        self.conv = conv
        self.max = max

    def forward(self, x):
        x = self.conv(x)
        return self.max(x), x, x.shape[2]


class Up(nn.Module):
    """
    Up blocks in U-Net

    Similar to the down blocks, but incorporates input from skip connections.
    """
    def __init__(self, up, conv):
        super(Up, self).__init__()
        self.conv = conv
        self.up = up

    def forward(self, x, conv_out, D):
        x = self.up(x)
        lower = int(0.5 * (D - x.shape[2]))
        upper = int(D - lower)
        conv_out_ = conv_out[:, :, lower:upper, lower:upper] # adjust to zero padding
        x = torch.cat([x, conv_out_], dim=1)
        return self.conv(x)


class Unet(nn.Module):

    def __init__(self, model_opts):
        self.opts = model_opts["unet_opts"]
        super(Unet, self).__init__()
        self.n_input_channels = self.opts["n_input_channels"]
        self.n_classes = self.opts["n_classes"]

        # down transformations
        max2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_1 = Down(self.conv_block(self.n_input_channels, 32), max2d)
        self.down_2 = Down(self.conv_block(32, 64), max2d)
        self.down_3 = Down(self.conv_block(64, 128), max2d)
        self.down_4 = Down(self.conv_block(128, 256), max2d)

        # midpoint
        self.conv5_block = self.conv_block(256, 512)

        # up transformations
        conv_tr = lambda x, y: nn.ConvTranspose2d(x, y, kernel_size=2, stride=2)
        self.up_1 = Up(conv_tr(512, 256), self.conv_block(512, 256))
        self.up_2 = Up(conv_tr(256, 128), self.conv_block(256, 128))
        self.up_3 = Up(conv_tr(128, 64), self.conv_block(128, 64))
        self.up_4 = Up(conv_tr(64, 32), self.conv_block(64, 32))

        # Final output
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=self.n_classes,
                                    kernel_size=1, padding=0, stride=1)


    def conv_block(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=True):
        """
        This is the main conv block for Unet. Two conv2d
        :param dim_in:
        :param dim_out:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param useBN:
        :param useGN:
        :return:
        """
        if self.opts["normalization_type"] == "BN":
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True),
            )
        elif self.opts["normalization_type"] == "GN":
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                #FIXME add num_groups as hyper param on json
                nn_utils.GroupNorm(dim_out),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True)
            )


    def forward(self, x):
        # down layers
        x, conv1_out, conv1_dim = self.down_1(x)
        x, conv2_out, conv2_dim = self.down_2(x)
        x, conv3_out, conv3_dim = self.down_3(x)
        x, conv4_out, conv4_dim = self.down_4(x)

        # Bottleneck
        x = self.conv5_block(x)

        # up layers
        x = self.up_1(x, conv4_out, conv4_dim)
        x = self.up_2(x, conv3_out, conv3_dim)
        x = self.up_3(x, conv2_out, conv2_dim)
        x = self.up_4(x, conv1_out, conv1_dim)
        return self.conv_final(x)


#Test with mock data
if __name__ == "__main__":
    # A full forward pass
    params = json.load(open(os.environ["PARAMS_PATH"], "r"))
    model_opts = params["model_opts"]
    im = torch.randn(1, 4, 240, 240)
    model = Unet(params)
    x = model(im)
    print(x.shape)
    del model
    del x
    # print(x.shape)
