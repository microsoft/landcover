from ServerModelsAbstract import BackendModel
import torch
import numpy as np
import torch.nn as nn
from training.pytorch.models.fusionnet import Fusionnet
from training.pytorch.models.unet import Unet
import os, json
from torch.autograd import Variable

def softmax(output):
    output_max = np.max(output, axis=2, keepdims=True)
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=2, keepdims=True)
    return exps/exp_sums

class multiclass_ce(nn.modules.Module):
    def __init__(self):
        super(multiclass_ce, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss(ignore_index = 0, size_average=True)

    def __call__(self,y_true, y_pred):
        loss = self.crossentropy(y_pred, y_true)
        return loss

class GroupParams(nn.Module):

    def __init__(self, model):
        super(GroupParams, self).__init__()
        self.gammas = nn.Parameter(torch.ones((1, 32, 1, 1)))
        self.betas = nn.Parameter(torch.zeros((1, 32, 1, 1)))
        self.model = model

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, conv1_out, conv1_dim = self.model.down_1(x)


        # gammas = np.zeros((1, 32, 1, 1))
        # gammas[0, :8, 0, 0] = self.gammas.detach().numpy()[0]
        # gammas[0, 8:16, 0, 0] = self.gammas.detach().numpy()[1]
        # gammas[0, 16:24, 0, 0] = self.gammas.detach().numpy()[2]
        # gammas[0, 24:32, 0, 0] = self.gammas.detach().numpy()[3]
        #
        # betas = np.zeros((1, 32, 1, 1))
        # betas[0, :8, 0, 0] = self.betas.detach().numpy()[0]
        # betas[0, 8:16, 0, 0] = self.betas.detach().numpy()[1]
        # betas[0, 16:24, 0, 0] = self.betas.detach().numpy()[2]
        # betas[0, 24:32, 0, 0] = self.betas.detach().numpy()[3]

        self.gammas.to(device)
        self.betas.to(device)


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
        x = x * self.gammas + self.betas

        return self.model.conv_final(x)

class UnetgnFineTune(BackendModel):

    def __init__(self, model_fn, gpuid):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.output_channels = 5
        self.input_size = 240
        self.model_fn = model_fn
        self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn8/training/params.json", "r"))["model_opts"]
        self.inf_framework = InferenceFramework(Unet, self.opts)
        self.inf_framework.load_model(self.model_fn)
        for param in self.inf_framework.model.parameters():
            param.requires_grad = False

        # ------------------------------------------------------
        # Step 2
        #   Pre-load augment model seed data
        # ------------------------------------------------------
        self.current_features = None

        self.augment_base_x_train = []
        self.augment_base_y_train = []

        self.augment_x_train = []
        self.augment_y_train = []
        self.augment_model = GroupParams(self.inf_framework.model)
        self.augment_model_trained = False

        seed_x_fn = ""
        seed_y_fn = ""
        # if superres:
        #     seed_x_fn = "data/seed_data_hr+sr_x.npy"
        #     seed_y_fn = "data/seed_data_hr+sr_y.npy"
        # else:
        #     seed_x_fn = "data/seed_data_hr_x.npy"
        #     seed_y_fn = "data/seed_data_hr_y.npy"
        # for row in np.load(seed_x_fn):
        #     self.augment_base_x_train.append(row)
        # for row in np.load(seed_y_fn):
        #     self.augment_base_y_train.append(row)
        #
        # for row in self.augment_base_x_train:
        #     self.augment_x_train.append(row)
        # for row in self.augment_base_y_train:
        #     self.augment_y_train.append(row)
        self.naip_data = None
        self.correction_labels = None
        self.tile_padding = 0

        self.down_weight_padding = 10

        self.stride_x = self.input_size - self.down_weight_padding * 2
        self.stride_y = self.input_size - self.down_weight_padding * 2
        self.batch_x = []
        self.batch_y = []
        self.num_corrected_pixels = 0
        self.batch_count = 0

    def run(self, naip_data, naip_fn, extent, padding):

        # apply padding to the output_features
        x=naip_data
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        x = np.rollaxis(x, 2, 1)
        x = x[:4, :, :]
        naip_data = x / 255.0
        output = self.run_model_on_tile(naip_data)
        if padding > 0:
            self.tile_padding = padding
          #  naip_data_trimmed = naip_data[:, padding:-padding, padding:-padding]
          #  output_trimmed = output[:, padding:-padding, padding:-padding]
        self.naip_data = naip_data  # keep non-trimmed size, i.e. with padding
        self.correction_labels = np.zeros((naip_data.shape[1], naip_data.shape[2], self.output_channels),
                                          dtype=np.float32)

        self.last_output = output
        return output

#FIXME: add retrain method
    def retrain(self, train_steps=100, corrections_from_ui=True, learning_rate=0.004):
        num_labels = np.count_nonzero(self.correction_labels)
        print("Fine tuning group norm params with %d new labels. 4 Groups, 8 Params" % num_labels)

        height = self.naip_data.shape[1]
        width = self.naip_data.shape[2]

        batch_x = []
        batch_y = []
        batch_count = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        number_corrected_pixels = 0.0

        if corrections_from_ui:
            correction_labels = self.correction_labels
        else:
            correction_labels = np.zeros(( self.last_output.shape[0], self.last_output.shape[1], 5))
            for i in range(correction_labels.shape[0]):
                for j in range(correction_labels.shape[1]):
                    label_index = self.last_output[i][j].argmax()
                    correction_labels[i, j, label_index + 1] = 1.0

        for y_index in (list(range(0, height - self.input_size, self.stride_y)) + [height - self.input_size, ]):
            for x_index in (list(range(0, width - self.input_size, self.stride_x)) + [width - self.input_size, ]):
                naip_im = self.naip_data[:, y_index:y_index + self.input_size, x_index:x_index + self.input_size]
                correction_labels_slice = correction_labels[y_index:y_index + self.input_size,
                                          x_index:x_index + self.input_size, :]
                # correction_labels = test_correction_labels[y_index:y_index+self.input_size, x_index:x_index+self.input_size, :]
        self.batch_x.append(batch_x)
        self.batch_y.append(batch_y)
        self.num_corrected_pixels += number_corrected_pixels
        self.batch_count += batch_count

        batch_arr_x = np.zeros((batch_count, 4, self.input_size, self.input_size))
        batch_arr_y = np.zeros((batch_count, self.input_size, self.input_size))
        i, j = 0, 0
        for im in batch_x:
            batch_arr_x[i, :, :, :] = im
            i += 1
        batch_x = torch.from_numpy(batch_arr_x).float().to(device)
        for y in batch_y:
            batch_arr_y[j, :, :] = np.argmax(y, axis=2)
            j += 1
        batch_y = torch.from_numpy(batch_arr_y).float().to(device)

        optimizer = torch.optim.Adam(self.augment_model.parameters(), lr=learning_rate, eps=1e-5)
        optimizer.zero_grad()
        criterion = multiclass_ce().to(device)
        # pdb.set_trace()

        for i in range(train_steps):
            with torch.set_grad_enabled(True):
                outputs = self.augment_model.forward(batch_x[:, :, 2:240 - 2, 2:240 - 2])
                loss = criterion(torch.squeeze(batch_y[:, 94:240 - 94, 94:240 - 94],1).long(), outputs)
                print(loss.item())
                loss.backward()
                optimizer.step()

        # pdb.set_trace()

        success = True
        message = "Fine-tuned Group norm params with %d samples. 4 Groups. 8 params, 1 layer." % num_labels

        return success, message

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        padding = self.tile_padding

        self.correction_labels[tdst_row + padding: bdst_row + 1 + padding,
        tdst_col + padding: bdst_col + 1 + padding, :] = 0.0
        self.correction_labels[tdst_row + padding: bdst_row + 1 + padding,
        tdst_col + padding: bdst_col + 1 + padding,
        class_idx + 1] = 1.0

    def reset(self):
        #self.augment_x_train = []
        #self.augment_y_train = []
        self.augment_model = GroupParams(self.inf_framework.model)
        self.augment_model_trained = False

        #for row in self.augment_base_x_train:
        #    self.augment_x_train.append(row)
        #for row in self.augment_base_y_train:
        #    self.augment_y_train.append(row)

    def run_model_on_tile(self, naip_tile, batch_size=32):
        y_hat = self.predict_entire_image_unet_fine(naip_tile)
        output = y_hat[:, :, 1:5]
        return softmax(output)

    def predict_entire_image_unet_fine(self, x):
        if torch.cuda.is_available():
            self.augment_model.cuda()
        norm_image = x
        _, w, h = norm_image.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = np.zeros((5, w, h))

        norm_image1 = norm_image[:, 130:w - (w % 892) + 130, 130:h - (h % 892) + 130]
        x_c_tensor1 = torch.from_numpy(norm_image1).float().to(device)
        y_pred1 = self.augment_model.forward(x_c_tensor1.unsqueeze(0))
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        out[:, 92 + 130:w - (w % 892) + 130 - 92, 92 + 130:h - (h % 892) - 92 + 130] = y_hat1
        pred = np.rollaxis(out, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        return pred




class GroupParamsFusionnet(nn.Module):

    def __init__(self, model):
        super(GroupParamsFusionnet, self).__init__()
        self.gammas = nn.Parameter(torch.ones((1, 32, 1, 1)))
        self.betas = nn.Parameter(torch.zeros((1, 32, 1, 1)))
       # self.gammas2 = nn.Parameter(torch.ones((1, 64, 1, 1)))
        #self.betas2 = nn.Parameter(torch.zeros((1, 64, 1, 1)))
        self.model = model

    def forward(self, input):

        down_1 = self.model.down_1(input)
        pool_1 = self.model.pool_1(down_1)
        #pool_1 = pool_1 * self.gammas + self.betas
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
        #up_3 = up_3 * self.gammas2 + self.betas2

        deconv_4 = self.model.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.model.up_4(skip_4)
        up_4 = up_4 * self.gammas + self.betas

        out = self.model.out(up_4)
        out = self.model.out_2(out)
        # out = torch.clamp(out, min=-1, max=1)
        return out


class FusionnetgnFineTune(BackendModel):

    def __init__(self, model_fn, gpuid):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.output_channels = 5
        self.input_size = 512
        self.model_fn = model_fn
        self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_fusionnet32_gn_8_isotropic/training/params.json", "r"))["model_opts"]
        self.inf_framework = InferenceFramework(Fusionnet, self.opts)
        self.inf_framework.load_model(self.model_fn)
        for param in self.inf_framework.model.parameters():
            param.requires_grad = False

        # ------------------------------------------------------
        # Step 2
        #   Pre-load augment model seed data
        # ------------------------------------------------------
        self.current_features = None

        self.augment_base_x_train = []
        self.augment_base_y_train = []

        self.augment_x_train = []
        self.augment_y_train = []
        self.augment_model = GroupParamsFusionnet(self.inf_framework.model)
        self.augment_model_trained = False

        seed_x_fn = ""
        seed_y_fn = ""
        # if superres:
        #     seed_x_fn = "data/seed_data_hr+sr_x.npy"
        #     seed_y_fn = "data/seed_data_hr+sr_y.npy"
        # else:
        #     seed_x_fn = "data/seed_data_hr_x.npy"
        #     seed_y_fn = "data/seed_data_hr_y.npy"
        # for row in np.load(seed_x_fn):
        #     self.augment_base_x_train.append(row)
        # for row in np.load(seed_y_fn):
        #     self.augment_base_y_train.append(row)
        #
        # for row in self.augment_base_x_train:
        #     self.augment_x_train.append(row)
        # for row in self.augment_base_y_train:
        #     self.augment_y_train.append(row)
        self.naip_data = None
        self.correction_labels = None
        self.tile_padding = 0

        self.down_weight_padding = 10

        self.stride_x = self.input_size - self.down_weight_padding * 2
        self.stride_y = self.input_size - self.down_weight_padding * 2
        self.batch_x = []
        self.batch_y = []
        self.num_corrected_pixels = 0
        self.batch_count = 0

    def run(self, naip_data, naip_fn, extent, padding):

        # apply padding to the output_features
        x=naip_data
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        x = np.rollaxis(x, 2, 1)
        x = x[:4, :, :]
        naip_data = x / 255.0
        output = self.run_model_on_tile(naip_data)
        if padding > 0:
            self.tile_padding = padding
          #  naip_data_trimmed = naip_data[:, padding:-padding, padding:-padding]
          #  output_trimmed = output[:, padding:-padding, padding:-padding]
        self.naip_data = naip_data  # keep non-trimmed size, i.e. with padding
        self.correction_labels = np.zeros((naip_data.shape[1], naip_data.shape[2], self.output_channels),
                                          dtype=np.float32)

        self.last_output = output
        return output

#FIXME: add retrain method
    def retrain(self, train_steps=10, corrections_from_ui=True, learning_rate=0.01):
        num_labels = np.count_nonzero(self.correction_labels)
        print("Fine-tuning Group norm params with %d new labels" % num_labels)

        height = self.naip_data.shape[1]
        width = self.naip_data.shape[2]

        batch_x = []
        batch_y = []
        batch_count = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        number_corrected_pixels = 0.0

        if corrections_from_ui:
            correction_labels = self.correction_labels
        else:
            correction_labels = np.zeros(( self.last_output.shape[0], self.last_output.shape[1], 5))
            for i in range(correction_labels.shape[0]):
                for j in range(correction_labels.shape[1]):
                    label_index = self.last_output[i][j].argmax()
                    correction_labels[i, j, label_index + 1] = 1.0

        for y_index in (list(range(0, height - self.input_size, self.stride_y)) + [height - self.input_size, ]):
            for x_index in (list(range(0, width - self.input_size, self.stride_x)) + [width - self.input_size, ]):
                naip_im = self.naip_data[:, y_index:y_index + self.input_size, x_index:x_index + self.input_size]
                correction_labels_slice = correction_labels[y_index:y_index + self.input_size,
                                          x_index:x_index + self.input_size, :]
                # correction_labels = test_correction_labels[y_index:y_index+self.input_size, x_index:x_index+self.input_size, :]

                batch_x.append(naip_im)
                batch_y.append(correction_labels_slice)

                batch_count += 1
                number_corrected_pixels += len(correction_labels_slice.nonzero()[0])

        self.batch_x.append(batch_x)
        self.batch_y.append(batch_y)
        self.num_corrected_pixels += number_corrected_pixels
        self.batch_count += batch_count

        batch_arr_x = np.zeros((batch_count, 4, self.input_size, self.input_size))
        batch_arr_y = np.zeros((batch_count, self.input_size, self.input_size))
        i, j = 0, 0
        for im in batch_x:
            batch_arr_x[i, :, :, :] = im
            i += 1
        batch_x = torch.from_numpy(batch_arr_x).float().to(device)
        for y in batch_y:
            batch_arr_y[j, :, :] = np.argmax(y, axis=2)
            j += 1
        batch_y = torch.from_numpy(batch_arr_y).float().to(device)

        optimizer = torch.optim.Adam(self.augment_model.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        criterion = multiclass_ce().to(device)
        # pdb.set_trace()

        for i in range(train_steps):
            with torch.set_grad_enabled(True):
                outputs = self.augment_model.forward(batch_x)
                loss = criterion(torch.squeeze(batch_y,1).long(), outputs)
                loss.backward()
                optimizer.step()

        # pdb.set_trace()

        success = True
        message = "Fine-tuned Group norm params with %d samples. 4 Groups. 8 params, 1 layer." % num_labels

        return success, message

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        padding = self.tile_padding

        self.correction_labels[tdst_row + padding: bdst_row + 1 + padding,
        tdst_col + padding: bdst_col + 1 + padding, :] = 0.0
        self.correction_labels[tdst_row + padding: bdst_row + 1 + padding,
        tdst_col + padding: bdst_col + 1 + padding,
        class_idx + 1] = 1.0

    def reset(self):
        #self.augment_x_train = []
        #self.augment_y_train = []
        self.augment_model = GroupParamsFusionnet(self.inf_framework.model)
        self.augment_model_trained = False

        #for row in self.augment_base_x_train:
        #    self.augment_x_train.append(row)
        #for row in self.augment_base_y_train:
        #    self.augment_y_train.append(row)

    def run_model_on_tile(self, naip_tile, batch_size=32):
        y_hat = self.predict_entire_image_fusionnet_fine(naip_tile)
        output = y_hat[:, :, 1:5]
        return softmax(output)

    def predict_entire_image_fusionnet_fine(self, x):
        self.augment_model.eval()
        if torch.cuda.is_available():
            self.augment_model.cuda()
        naip_tile = x

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
            batch_arr[i, :, :, :] = im
            i += 1
        batch = torch.from_numpy(batch_arr).float().to(device)
        model_output = self.augment_model.forward(batch)
        model_output = (Variable(model_output).data).cpu().numpy()
        for i, (y, x) in enumerate(batch_indices):
            output[:, y:y + self.input_size, x:x + self.input_size] += model_output[i] * kernel[np.newaxis, ...]
            counts[y:y + self.input_size, x:x + self.input_size] += kernel

        output = output / counts[np.newaxis, ...]
        pred = np.rollaxis(output, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        return pred

class InferenceFramework():
    def __init__(self, model, opts):
        self.opts = opts
        self.model = model(self.opts)

    def load_model(self, path_2_saved_model):
        checkpoint = torch.load(path_2_saved_model)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()



