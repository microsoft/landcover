from web_tool.ServerModelsAbstract import BackendModel
import torch
import numpy as np
import torch.nn as nn
import copy
import os, json
from training.pytorch.utils.eval_segm import mean_IoU, pixel_accuracy
from training.pytorch.models.unet import Unet
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
        try:
            self.border_margin_px = model.border_margin_px
        except:
            self.border_margin_px = 0

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
        x = x * self.gammas.to(device) + self.betas.to(device)

        return self.model.conv_final(x)
    
class UnetgnFineTune(BackendModel):

    def __init__(self, model_fn, gpuid):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.output_channels = 5
        self.input_size = 240
        self.model_fn = model_fn
        self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/params.json", "r"))["model_opts"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.init_model()
        self.model_trained = False
        self.naip_data = None
        self.correction_labels = None
        self.tile_padding = 0

        self.down_weight_padding = 40

        self.stride_x = self.input_size - self.down_weight_padding * 2
        self.stride_y = self.input_size - self.down_weight_padding * 2
        self.batch_x = []
        self.batch_y = []
        self.num_corrected_pixels = 0
        self.batch_count = 0
        self.run_done = False
        self.rows = 892
        self.cols = 892

    def run(self, naip_data, naip_fn, extent, padding):
        if self.correction_labels is not None:
            self.set_corrections()

        # apply padding to the output_features
        x=naip_data
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        x = x[:4, :, :]
        naip_data = x / 255.0
        output = self.run_model_on_tile(naip_data)
        if padding > 0:
            self.tile_padding = padding
        self.naip_data = naip_data  # keep non-trimmed size, i.e. with padding
        self.correction_labels = np.zeros((naip_data.shape[1], naip_data.shape[2], self.output_channels),
                                          dtype=np.float32)
        self.last_output = output
        return output

    def set_corrections(self):
        num_labels = np.count_nonzero(self.correction_labels)
        batch_count = 0
        correction_labels = self.correction_labels

        batch_xi = self.naip_data[:, 130:self.rows + 130, 130:self.cols + 130]
        batch_yi = np.argmax(correction_labels[130:self.rows + 130, 130:self.cols + 130, :], axis=2)
        if(num_labels>0):
            self.batch_x.append(batch_xi)
            self.batch_y.append(batch_yi)
            self.num_corrected_pixels += num_labels
            self.batch_count += batch_count

    def retrain(self, train_steps=25, learning_rate=0.0015):
        #if self.batch_count != 0 and self.correction_labels is not None:
        self.set_corrections()
        print_every_k_steps = 1

        print("Fine tuning group norm params with %d new labels. 4 Groups, 8 Params" % self.num_corrected_pixels)
        batch_x = np.array(self.batch_x)
        number_windows, channels, rows , cols = batch_x.shape
        batch_y = np.array(self.batch_y)
        batch_y = torch.from_numpy(batch_y).float().to(self.device)
        self.init_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        #optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=4, history_size=7)
        optimizer.zero_grad()
        criterion = multiclass_ce().to(self.device)

        for i in range(train_steps):
           # print('step %d' % i)
            iou = 0
            acc = 0
            for j in range(number_windows):
                with torch.set_grad_enabled(True):
                    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    out = torch.zeros((5, self.rows, self.cols))
                    x_c_tensor1 = torch.from_numpy(batch_x[j]).float().to(self.device)
                    y_pred1 = self.model.forward(x_c_tensor1.unsqueeze(0))
                    out[:, 92: -92, 92:-92] = y_pred1
                    outputs = out.float().to(self.device)
                    y_hat1 = (Variable(out).data).cpu().numpy()
                    y_hat1 = np.argmax(y_hat1, axis=0)
                    y_true = (Variable(batch_y[j]).data).cpu().numpy()
                    # iou+=mean_IoU(y_hat1, y_true,{0})
                    acc += pixel_accuracy(y_hat1, y_true, {0})
                    loss = criterion(torch.unsqueeze(batch_y[j], 0).long(), torch.unsqueeze(outputs, 0))
                    loss.backward()
                    #def closure():
                        #out[:, 92: -92, 92:-92] = y_pred1
                        #outputs = out.float().to(device)
                       # loss = criterion(torch.unsqueeze(batch_y[j],0).long(), torch.unsqueeze(outputs,0))
                       # loss.backward(retain_graph=True)
                       # return loss
                   # optimizer.step(closure)
                    optimizer.step()
            iou/=number_windows
            acc/=number_windows
            if i % print_every_k_steps == 0:
                print("Step pixel acc: ", acc)

        success = True
        message = "Fine-tuned Group norm params with %d samples. 4 Groups. 8 params, 1 layer." % self.num_corrected_pixels
        print(message)
        return success, message

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
       # pdb.set_trace()
        padding = self.tile_padding
        print("adding sample: class %d (incremented to %d) at (%d, %d)" % (class_idx, class_idx + 1, tdst_row, tdst_row))
        self.correction_labels[tdst_row + padding: bdst_row + 1 + padding,
                               tdst_col + padding: bdst_col + 1 + padding, :] = 0.0
        self.correction_labels[tdst_row + padding: bdst_row + 1 + padding,
                               tdst_col + padding: bdst_col + 1 + padding, class_idx + 1] = 1.0


    def init_model(self):
        self.model = GroupParams(self.inf_framework.model)
        self.model.to(self.device)
        
    def reset(self):
        self.init_model()
        self.model_trained = False
        self.batch_x = []
        self.batch_y = []
        self.run_done = False
        self.num_corrected_pixels = 0

    def run_model_on_tile(self, naip_tile, batch_size=32):
        y_hat = self.predict_entire_image_unet_fine(naip_tile)
        output = y_hat[:, :, 1:5]
        return softmax(output)

    def predict_entire_image_unet_fine(self, x):
        if torch.cuda.is_available():
            self.model.cuda()
        norm_image = x
        _, w, h = norm_image.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = np.zeros((5, w, h))

        norm_image1 = norm_image[:, 130:w - (w % 892) + 130, 130:h - (h % 892) + 130]
        x_c_tensor1 = torch.from_numpy(norm_image1).float().to(device)
        y_pred1 = self.model.forward(x_c_tensor1.unsqueeze(0))
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        out[:, 92 + 130:w - (w % 892) + 130 - 92, 92 + 130:h - (h % 892) - 92 + 130] = y_hat1
        pred = np.rollaxis(out, 0, 3)
        print(pred.shape)
        return pred


class LastKLayersFineTune(UnetgnFineTune):

    def __init__(self, model_fn, gpuid, last_k_layers=1):
        super().__init__(model_fn, gpuid)
        self.old_inference_framework = copy.deepcopy(self.inf_framework)
        self.last_k_layers = last_k_layers
        print('in LastKLayersFineTune init')
        self.init_model()

    def init_model(self):
        try:
            self.inf_framework = copy.deepcopy(self.old_inference_framework)
            self.model = self.inf_framework.model
            self.model.to(self.device)

            k = self.last_k_layers

            # Freeze all but last k layers
            for layer in list(self.model.children())[:-k]:
                for param in layer.parameters():
                    param.requires_grad = False

            # Un-freeze last k layers
            for layer in list(self.model.children())[-k:]:
                for param in layer.parameters():
                    param.requires_grad = True
        except:
            print("Trying to copy inf_framework before it exists")


class GroupParamsLastKLayersFineTune(UnetgnFineTune):

    def __init__(self, model_fn, gpuid, last_k_layers=1):
        super().__init__(model_fn, gpuid)
        self.old_inference_framework = copy.deepcopy(self.inf_framework)
        self.last_k_layers = last_k_layers
        self.init_model()

    def init_model(self):
        try:
            self.inf_framework = copy.deepcopy(self.old_inference_framework)
            k = self.last_k_layers

            # Freeze all but last k layers
            for layer in list(self.model.children())[:-k]:
                for param in layer.parameters():
                    param.requires_grad = False

            # Un-freeze last k layers
            for layer in list(self.model.children())[-k:]:
                for param in layer.parameters():
                    param.requires_grad = True

            self.model = GroupParams(self.inf_framework.model)
            self.model.to(self.device)

        except:
            print("Trying to copy inf_framework before it exists")

class GroupParamsThenLastKLayersFineTune(UnetgnFineTune):

    def __init__(self, model_fn, gpuid, last_k_layers=1):
        super().__init__(model_fn, gpuid)
        self.old_inference_framework = copy.deepcopy(self.inf_framework)
        self.last_k_layers = last_k_layers
        self.init_model()

    def retrain(self, train_steps=8, learning_rate=0.0015):
        #if self.batch_count != 0 and self.correction_labels is not None:
        self.set_corrections()
        print_every_k_steps = 1
        k = self.last_k_layers

        print("Fine tuning group norm params with %d new labels. 4 Groups, 8 Params" % self.num_corrected_pixels)
        batch_x = np.array(self.batch_x)
        number_windows, channels, rows, cols = batch_x.shape
        batch_y = np.array(self.batch_y)
        batch_y = torch.from_numpy(batch_y).float().to(self.device)
        self.init_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        # optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=4, history_size=7)
        optimizer.zero_grad()
        criterion = multiclass_ce().to(self.device)

        for i in range(train_steps):
            # print('step %d' % i)
            iou = 0
            acc = 0
            for j in range(number_windows):
                with torch.set_grad_enabled(True):
                    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    out = torch.zeros((5, self.rows, self.cols))
                    x_c_tensor1 = torch.from_numpy(batch_x[j]).float().to(self.device)
                    y_pred1 = self.model.forward(x_c_tensor1.unsqueeze(0))
                    out[:, 92: -92, 92:-92] = y_pred1
                    outputs = out.float().to(self.device)
                    y_hat1 = (Variable(out).data).cpu().numpy()
                    y_hat1 = np.argmax(y_hat1, axis=0)
                    y_true = (Variable(batch_y[j]).data).cpu().numpy()
                    # iou+=mean_IoU(y_hat1, y_true,{0})
                    acc += pixel_accuracy(y_hat1, y_true, {0})
                    loss = criterion(torch.unsqueeze(batch_y[j], 0).long(), torch.unsqueeze(outputs, 0))
                    loss.backward()
                    # def closure():
                    # out[:, 92: -92, 92:-92] = y_pred1
                    # outputs = out.float().to(device)
                    # loss = criterion(torch.unsqueeze(batch_y[j],0).long(), torch.unsqueeze(outputs,0))
                    # loss.backward(retain_graph=True)
                    # return loss
                    # optimizer.step(closure)
                    optimizer.step()
            iou /= number_windows
            acc /= number_windows
            if i % print_every_k_steps == 0:
                print("Step pixel acc: ", acc)

        for layer in list(self.model.children())[:-k]:
            for param in layer.parameters():
                param.requires_grad = False

        # Un-freeze last k layers
        for layer in list(self.model.children())[-k:]:
            for param in layer.parameters():
                param.requires_grad = True

        for i in range(int(train_steps/2)):
            # print('step %d' % i)
            iou = 0
            acc = 0
            for j in range(number_windows):
                with torch.set_grad_enabled(True):
                    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    out = torch.zeros((5, self.rows, self.cols))
                    x_c_tensor1 = torch.from_numpy(batch_x[j]).float().to(self.device)
                    y_pred1 = self.model.forward(x_c_tensor1.unsqueeze(0))
                    out[:, 92: -92, 92:-92] = y_pred1
                    outputs = out.float().to(self.device)
                    y_hat1 = (Variable(out).data).cpu().numpy()
                    y_hat1 = np.argmax(y_hat1, axis=0)
                    y_true = (Variable(batch_y[j]).data).cpu().numpy()
                    # iou+=mean_IoU(y_hat1, y_true,{0})
                    acc += pixel_accuracy(y_hat1, y_true, {0})
                    loss = criterion(torch.unsqueeze(batch_y[j], 0).long(), torch.unsqueeze(outputs, 0))
                    loss.backward()
                    # def closure():
                    # out[:, 92: -92, 92:-92] = y_pred1
                    # outputs = out.float().to(device)
                    # loss = criterion(torch.unsqueeze(batch_y[j],0).long(), torch.unsqueeze(outputs,0))
                    # loss.backward(retain_graph=True)
                    # return loss
                    # optimizer.step(closure)
                    optimizer.step()
            iou /= number_windows
            acc /= number_windows
            if i % print_every_k_steps == 0:
                print("Step pixel acc: ", acc)

        success = True
        message = "Fine-tuned Group norm params with %d samples. 4 Groups. 8 params, 1 layer." % self.num_corrected_pixels
        print(message)
        return success, message


    def reset(self):
        self.inf_framework = InferenceFramework(Unet, self.opts)
        self.inf_framework.load_model(self.model_fn)
        for param in self.inf_framework.model.parameters():
            param.requires_grad = False
        self.init_model()
        self.model_trained = False
        self.batch_x = []
        self.batch_y = []
        self.run_done = False
        self.num_corrected_pixels = 0


class InferenceFramework():
    def __init__(self, model, opts):
        self.opts = opts
        self.model = model(self.opts)

    def load_model(self, path_2_saved_model):
        checkpoint = torch.load(path_2_saved_model)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()



