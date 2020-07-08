import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Unet model definition. 

Code mostly taken from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py
"""


class Unet(nn.Module):

    def __init__(self, feature_scale=1,
                 n_classes=3, in_channels=3,
                 is_deconv=True, is_batchnorm=False):
        """

        Args:
            feature_scale: the smallest number of filters (depth c) is 64 when feature_scale is 1,
                           and it is 32 when feature_scale is 2
            n_classes: number of output classes
            in_channels: number of channels in input
            is_deconv:
            is_batchnorm:
        """

        super(Unet, self).__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        assert 64 % self.feature_scale == 0, f'feature_scale {self.feature_scale} does not work with this UNet'

        filters = [64, 128, 256, 512, 1024]  # this is `c` in the diagram, [c, 2c, 4c, 8c, 16c]
        filters = [int(x / self.feature_scale) for x in filters]
        logging.info('filters used are: {}'.format(filters))

        # downsampling
        self.conv1 = UnetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UnetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
    
    def forward_features(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final, up1


class UnetConv2(nn.Module):

    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(UnetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                # this amount of padding/stride/kernel_size preserves width/height
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp(nn.Module):

    def __init__(self, in_channels, out_channels, is_deconv):
        """

        is_deconv:  use transposed conv layer to upsample - parameters are learnt; otherwise use
                    bilinear interpolation to upsample.
        """
        super(UnetUp, self).__init__()

        self.conv = UnetConv2(in_channels, out_channels, False)

        self.is_deconv = is_deconv
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # UpsamplingBilinear2d is deprecated in favor of interpolate()
        # else:
        #     self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        """
        inputs1 is from the downward path, of higher resolution
        inputs2 is from the 'lower' layer. It gets upsampled (spatial size increases) and its depth (channels) halves
        to match the depth of inputs1, before being concatenated in the depth dimension.
        """
        if self.is_deconv:
            outputs2 = self.up(inputs2)
        else:
            # scale_factor is the multiplier for spatial size
            outputs2 = F.interpolate(inputs2, scale_factor=2, mode='bilinear', align_corners=True)

        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], dim=1))
