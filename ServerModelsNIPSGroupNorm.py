from ServerModelsAbstract import BackendModel
import torch
import numpy as np
from fusionnet import Fusionnet
from unet import Unet
import os, json

def softmax(output):
    output_max = np.max(output, axis=2, keepdims=True)
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=2, keepdims=True)
    return exps/exp_sums

class UnetgnFineTune(BackendModel):

    def __init__(self, model_fn, gpuid, superres=False):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.model_fn = model_fn
        self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_conditional_superres512/training/params.json", "r"))["model_opts"]

    def run(self, naip_data, naip_fn, extent, padding):
        return self.run_model_on_tile(naip_data), os.path.basename(self.model_fn)

    def retrain(self):
        return

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        return

    def reset(self):
        return

    def run_model_on_tile(self, naip_tile, batch_size=32):
        inf_framework = InferenceFramework(Fusionnet, self.opts)
        inf_framework.load_model(self.model_fn)
        y_hat = inf_framework.predict_entire_image_fusionnet(naip_tile)
        output = y_hat[:, :, 1:5]
        return softmax(output)

class FusionnetgnFineTune(BackendModel):

    def __init__(self, model_fn, gpuid, superres=False):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.model_fn = model_fn
        self.opts = \
        json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn/training/params.json", "r"))[
            "model_opts"]

    def run(self, naip_data, naip_fn, extent, padding):
        return self.run_model_on_tile(naip_data), os.path.basename(self.model_fn)

    def retrain(self):
        return

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        return

    def reset(self):
        return

    def run_model_on_tile(self, naip_tile, batch_size=32):
        inf_framework = InferenceFramework(Unet, self.opts)
        inf_framework.load_model(self.model_fn)
        y_hat = inf_framework.predict_entire_image_unet(naip_tile)
        output = y_hat[:, :, 1:5]
        return softmax(output)




class InferenceFramework():
    def __init__(self, model, opts):
        self.opts = opts
        self.model = model(self.opts)
        self.output_channels = 5
        self.input_size = 512


    def load_model(self, path_2_saved_model):
        checkpoint = torch.load(path_2_saved_model)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()


    def fusionnet_gn_fun(self, x):

        down_1 = self.model.down_1(x)
        pool_1 = self.model.pool_1(down_1)
        down_2 = self.model.down_2(pool_1)
        pool_2 = self.model.pool_2(down_2)
        down_3 = self.model.down_3(pool_2)
        pool_3 = self.model.pool_3(down_3)
        down_4 = self.model.down_4(pool_3)
        pool_4 = self.model.pool_4(down_4)

        bridge = self.model.bridge(pool_4)

        deconv_1 = self.model.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.model.up_1(skip_1)
        deconv_2 = self.model.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.model.up_2(skip_2)
        deconv_3 = self.model.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.model.up_3(skip_3)
        deconv_4 = self.model.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.model.up_4(skip_4)

     #   gammas = np.zeros((1, 32, 1, 1))
    #    gammas[0, :8, 0, 0] = gamma[0]
    #    gammas[0, 8:16, 0, 0] = gamma[1]
   #     gammas[0, 16:24, 0, 0] = gamma[2]
   #     gammas[0, 24:32, 0, 0] = gamma[3]

    #    betas = np.zeros((1, 32, 1, 1))
   #     betas[0, :8, 0, 0] = beta[0]
   #     betas[0, 8:16, 0, 0] = beta[1]
    #    betas[0, 16:24, 0, 0] = beta[2]
    #    betas[0, 24:32, 0, 0] = beta[3]
    #    for id in dropouts:
    #        gammas[0, id, 0, 0] = 0
    #    gammas = torch.Tensor(gammas).to('cuda')
    #    betas = torch.Tensor(betas).to('cuda')
    #    up_4 = up_4 * gammas + betas

        out = self.model.out(up_4)

        #out = self.model.out_2(out)

        return out

    def unet_gn_fun(self, x):

        x, conv1_out, conv1_dim = self.model.down_1(x)

        # gammas = np.zeros((1, 32, 1, 1))
        # gammas[0, :8, 0, 0] = gamma[0]
        # gammas[0, 8:16, 0, 0] = gamma[1]
        # gammas[0, 16:24, 0, 0] = gamma[2]
        # gammas[0, 24:32, 0, 0] = gamma[3]
        #
        # betas = np.zeros((1, 32, 1, 1))
        # betas[0, :8, 0, 0] = beta[0]
        # betas[0, 8:16, 0, 0] = beta[1]
        # betas[0, 16:24, 0, 0] = beta[2]
        # betas[0, 24:32, 0, 0] = beta[3]
        #
        # for id in dropouts:
        #     gammas[0, id, 0, 0] = 0
        # gammas = torch.Tensor(gammas).to('cuda')
        # betas = torch.Tensor(betas).to('cuda')
        # x = x * gammas + betas

        x, conv2_out, conv2_dim = self.model.down_2(x)
        x, conv3_out, conv3_dim = self.model.down_3(x)
        x, conv4_out, conv4_dim = self.model.down_4(x)

        # Bottleneck
        x = self.model.conv5_block(x)

        # up layers
        x = self.model.up_1(x, conv4_out, conv4_dim)
        x = self.model.up_2(x, conv3_out, conv3_dim)
        x = self.model.up_3(x, conv2_out, conv2_dim)
        x = self.model.up_4(x, conv1_out, conv1_dim)

        return self.model.conv_final(x)

    def predict_entire_image_unet(self, x):
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        if torch.cuda.is_available():
            self.model.cuda()
        x = np.rollaxis(x, 2, 1)
        x = x[:4, :, :]
        norm_image = x / 255.0
        _, w, h = norm_image.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = np.zeros((5, w, h))

        norm_image1 = norm_image[:, 130:w - (w % 892) + 130, 130:h - (h % 892) + 130]
        x_c_tensor1 = torch.from_numpy(norm_image1).float().to(device)
        y_pred1 = self.unet_gn_fun(x_c_tensor1.unsqueeze(0))
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        out[:, 92 + 130:w - (w % 892) + 130 - 92, 92 + 130:h - (h % 892) - 92 + 130] = y_hat1
        pred = np.rollaxis(out, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        return pred

    def predict_entire_image_fusionnet(self, x):
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        if torch.cuda.is_available():
            self.model.cuda()
        x = np.rollaxis(x, 2, 1)
        x = x[:4, :, :]
        naip_tile = x / 255.0

        down_weight_padding = 40
        height = naip_tile.shape[1]
        width = naip_tile.shape[2]

        stride_x = self.input_size - down_weight_padding * 2
        stride_y = self.input_size - down_weight_padding * 2

        output = np.zeros((self.output_channels, height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
        kernel = np.ones((self.input_size, self.input_size), dtype=np.float32) * 0.1
        kernel[20:-20, 20:-20] = 1
        kernel[down_weight_padding:down_weight_padding + stride_y,
        down_weight_padding:down_weight_padding + stride_x] = 5

        batch = []
        batch_indices = []

        batch_count = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for y_index in (list(range(0, height - self.input_size, stride_y)) + [height - self.input_size, ]):
            for x_index in (list(range(0, width - self.input_size, stride_x)) + [width - self.input_size, ]):
                naip_im = naip_tile[:, y_index:y_index + self.input_size, x_index:x_index + self.input_size]
                batch.append(naip_im)
                batch_indices.append((y_index, x_index))
                batch_count += 1
        batch_arr = np.zeros((batch_count, 4, self.input_size, self.input_size))
        i = 0
        for im in batch:
            batch_arr[i,:,:,:] = im
            i+=1
        batch = torch.from_numpy(batch_arr).float().to(device)
        model_output = self.fusionnet_gn_fun(batch)
        model_output = (Variable(model_output).data).cpu().numpy()
        for i, (y, x) in enumerate(batch_indices):
            output[:,y:y + self.input_size, x:x + self.input_size] += model_output[i] * kernel[np.newaxis, ...]
            counts[y:y + self.input_size, x:x + self.input_size] += kernel

        output = output / counts[np.newaxis, ...]
        pred = np.rollaxis(output, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        return pred


