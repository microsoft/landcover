import torch
import torch.nn as nn
from pytorch.models.conditioning_nlcd import Conditioning_nlcd
import json, os
import torch.nn.functional as F
import pytorch.utils.pytorch_model_utils as nn_utils
from pytorch.utils.fusionnet_blocks import *

"""
@uthor: Anthony Ortiz
Date: 03/25/2019
Last Modified: 03/25/2019
"""



class Conv_residual_conv(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input, gamma, beta):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * conv_3 + beta

        out = F.relu(out)

        return out


class Conditional_superres_net(nn.Module):

    def __init__(self, model_opts, n_embedding_units=15 * 15 * 512):
        super(Conditional_superres_net, self).__init__()

        self.opts = model_opts["conditional_superres_net_opts"]
        self.n_input_channels = self.opts["n_input_channels"]
        self.n_classes = self.opts["n_classes"]
        self.out_dim = self.opts["n_filters"]
        self.final_out_dim = self.opts["n_classes"]

        self.n_embedding_units_cbn = n_embedding_units
        self.n_hidden_cbn = self.opts["n_hidden_cbn"]
        self.n_features_cbn = self.opts["n_features_cbn"]

        # this predict latlong
        self.conditioning_model = Conditioning_nlcd(model_opts)
        if self.opts["end_to_end"]:
            self.conditioning_model.train()
        else:
            checkpoint = torch.load(self.opts["conditioning_net_ckpt_path"])
            self.conditioning_model.load_state_dict(checkpoint['model'])
            self.conditioning_model.eval()

        # MLP used to predict betas and gammas
        self.fc_cbn = nn.Sequential(
            nn.Linear(self.n_embedding_units_cbn, self.n_hidden_cbn),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden_cbn, 2 * self.n_features_cbn),
        )
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()


        # encoder

        self.down_1 = Conv_residual_conv(self.n_input_channels, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # bridge

        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder

        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        # output

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Tanh()
        '''
        self.out = nn.Sequential(
            nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(self.final_out_dim),
            nn.Tanh(),
        )
        '''

        # initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        conditioning_pred = self.conditioning_model(x)
        conditioning_info = self.conditioning_model.pre_pred(x)

        cbn = self.fc_cbn(conditioning_info)
        gammas = cbn[:, :int(cbn.shape[1] / 2)]
        betas = cbn[:, int(cbn.shape[1] / 2):]

        down_1 = self.down_1(x, gammas[:, :32], betas[:, :32])
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1, gammas[:, 32:96], betas[:, 32:96])
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2, gammas[:, 96:224], betas[:, 96:224])
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3, gammas[:, 224:480], betas[:, 224:480])
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4, gammas[:, 480:992], betas[:, 480:992])

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.up_1(skip_1, gammas[:, 992:1248], betas[:, 992:1248])
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.up_2(skip_2, gammas[:, 1248:1376], betas[:, 1248:1376])
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.up_3(skip_3, gammas[:, 1376:1440], betas[:, 1376:1440])
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.up_4(skip_4, gammas[:, 1440:1472], betas[:, 1440:1472])

        out = self.out(up_4)
        out = self.out_2(out)
        # out = torch.clamp(out, min=-1, max=1)
        if self.opts["end_to_end"]:
            return out, conditioning_pred
        return out



#Test with mock data
if __name__ == "__main__":
    # A full forward pass
    params = json.load(open(os.environ["PARAMS_PATH"], "r"))
    model_opts = params["model_opts"]
    im = torch.randn(1, 4, 240, 240)
    model = Conditional_superres_net(model_opts)
    x, latlon_pred = model(im)
    #x = model(im)
    print(x.shape)
    del model
    del x