import numpy as np
import torch
from msr_unet import Unet
import os
from torch.autograd import Variable
import torch.nn as nn

def normalize(img, percentile):
    '''
    Scale an image's 1 - percentile percentiles into 0 - 1 for display
    :param img:
    :return:
    '''
    orig_shape = img.shape
    if len(orig_shape) == 3:
        img = np.reshape(img,
                         [orig_shape[0] * orig_shape[1], orig_shape[2]]
                         ).astype(np.float32)
    elif len(orig_shape) == 2:
        img = np.reshape(img, [orig_shape[0] * orig_shape[1]]).astype(np.float32)
    mins = np.percentile(img, 1, axis = 0)
    maxs = np.percentile(img, percentile, axis = 0) - mins
    print(maxs)
    img = (img - mins) / maxs

    img.clip(0., 1.)
    img = np.reshape(img, orig_shape)

    return img

class InferenceFramework():
    def __init__(self, model):
        self.model = model()


    def load_model(self, path_2_saved_model):
        self.model.load_state_dict(torch.load(path_2_saved_model))
        self.model.eval()

    def predict_entire_image(self, x):
        if torch.cuda.is_available():
            self.model.cuda()
        print(x.shape)
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        print(x.shape)
        x = x[:3, :, :] / 255.0
        print(x.min(), x.max())
        #norm_image = normalize(x, 95)
        norm_image = x
        _, w, h = norm_image.shape
        out = np.zeros((4,w,h))
        norm_image1 = norm_image[:, :w-(w%128), :h-(h%128)]
        norm_image2 = norm_image[:, (w % 128):w, (h % 128):h ]
        norm_image3 = norm_image[:, :w - (w % 128), (h % 128):h]
        norm_image4 = norm_image[:, (w % 128):w, :h - (h % 128)]
        x_c_tensor1 = torch.from_numpy(norm_image1).float().cuda()
        x_c_tensor2 = torch.from_numpy(norm_image2).float().cuda()
        x_c_tensor3 = torch.from_numpy(norm_image3).float().cuda()
        x_c_tensor4 = torch.from_numpy(norm_image4).float().cuda()
        y_pred1 = self.model.forward(x_c_tensor1.unsqueeze(0))
        y_pred2 = self.model.forward(x_c_tensor2.unsqueeze(0))
        y_pred3 = self.model.forward(x_c_tensor3.unsqueeze(0))
        y_pred4 = self.model.forward(x_c_tensor4.unsqueeze(0))
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        y_hat2 = (Variable(y_pred2).data).cpu().numpy()
        y_hat3 = (Variable(y_pred3).data).cpu().numpy()
        y_hat4 = (Variable(y_pred4).data).cpu().numpy()
        out[:, :w - (w % 128), :h - (h % 128)] = y_hat1
        out[:, (w % 128):w, (h % 128):h ] = y_hat2
        out[:, :w - (w % 128), (h % 128):h] = y_hat3
        out[:, (w % 128):w, :h - (h % 128)] = y_hat4
        #out = np.rollaxis(out, 2, 1)
        out = np.rollaxis(out, 0, 3)
        print(out.shape)
        return out

def run(naip_data, naip_fn, extent, padding):
    output, name = run_cnn(naip_data)

    output = output[padding:-padding, padding:-padding, :]

    return output, name

def run_cnn(naip):
    y_hat = inf_framework.predict_entire_image(naip)
    return y_hat, "Torch Model"


inf_framework = InferenceFramework(Unet)
inf_framework.load_model("data/checkpoint-28.pt")