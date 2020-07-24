
from web_tool.ServerModelsAbstract import BackendModel
import torch as T
import numpy as np
import torch.nn as nn
import copy
import os, json
from training.pytorch.utils.eval_segm import mean_IoU, pixel_accuracy
from torch.autograd import Variable
import time
from scipy.special import softmax

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = T.nn.Conv2d(4,64,3,1,1)
        self.conv2 = T.nn.Conv2d(64,64,3,1,1)
        self.conv3 = T.nn.Conv2d(64,64,3,1,1)
        self.conv4 = T.nn.Conv2d(64,64,3,1,1)
        self.conv5 = T.nn.Conv2d(64,64,3,1,1)
        self.last = T.nn.Conv2d(64,22,1,1,0)
       
    def forward(self,inputs,prev_layer=False):
        x = T.relu(self.conv1(inputs))
        x = T.relu(self.conv2(x))
        x = T.relu(self.conv3(x))
        x = T.relu(self.conv4(x))
        x = T.relu(self.conv5(x))
        y = self.last(x)
        if prev_layer: return y,x
        else: return y
    
class TorchSmoothingFineTune(BackendModel):

    def __init__(self, model_fn, gpu_id, fine_tune_layer):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.output_channels = 22
        self.input_size = 240
        self.did_correction = False
        self.model_fn = model_fn
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.model = Model()
        self.init_model()
        for param in self.model.last.parameters():
            param.requires_grad = True
        print(sum(x.numel() for x in self.model.parameters()))

        # ------------------------------------------------------
        # Step 2
        #   Pre-load augment model seed data
        # ------------------------------------------------------
        self.current_features = None

        self.augment_base_x_train = []
        self.augment_base_y_train = []

        self.augment_x_train = []
        self.augment_y_train = []
        self.model_trained = False
        self.naip_data = None
        
        self.corr_features = []
        self.corr_labels = []
        
        self.num_corrected_pixels = 0
        self.batch_count = 0
        self.run_done = False
        self.rows = 892
        self.cols = 892

    def run(self, naip_data, naip_fn, extent):
       

        print(naip_data.shape)
      
        x = naip_data
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        x = x[:4, :, :]
        naip_data = x / 255.0
        output,features = self.run_model_on_tile(naip_data,True)
        
        
        self.features = features[0].cpu().numpy()
        self.naip_data = naip_data  # keep non-trimmed size, i.e. with padding
       
        self.last_output = output
        return output

    def retrain(self, train_steps=100, learning_rate=1e-3):
      
        print_every_k_steps = 10

        print("Fine tuning with %d new labels." % self.num_corrected_pixels)
        batch_x = T.from_numpy(np.array(self.corr_features)).float().to(self.device)
        batch_y = T.from_numpy(np.array(self.corr_labels)).to(self.device)
        
        self.init_model()
        
        optimizer = T.optim.Adam(self.model.last.parameters(), lr=learning_rate, eps=1e-5)
        
        criterion = T.nn.CrossEntropyLoss().to(self.device)

        for i in range(train_steps):
            #print('step %d' % i)
            acc = 0
            
            with T.enable_grad():

                optimizer.zero_grad()
                
                pred = self.model.last.forward(batch_x.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
                
                loss = criterion(pred,batch_y)
                
                print(loss.mean().item())
                
                acc = (pred.argmax(1)==batch_y).float().mean().item()

                loss.backward()
                optimizer.step()
            
            if i % print_every_k_steps == 0:
                print("Step pixel acc: ", acc)

        success = True
        message = "Fine-tuned model with %d samples." % len(self.corr_features)
        print(message)
        return success, message
    
    def undo(self):
        if len(self.corr_features)>0:
            self.corr_features = self.corr_features[:-1]
            self.corr_labels = self.corr_labels[:-1]
        print('undoing; now there are %d samples' % len(self.corr_features))

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        print("adding sample: class %d (incremented to %d) at (%d, %d)" % (class_idx, class_idx+1 , tdst_row, tdst_col))

        for i in range(tdst_row,bdst_row+1):
            for j in range(tdst_col,bdst_col+1):
                self.corr_labels.append(class_idx+1)
                self.corr_features.append(self.features[:,i,j])

    def init_model(self):
        checkpoint = T.load(self.model_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.to(self.device)
        
    def reset(self):
        self.init_model()
        self.model_trained = False
        self.batch_x = []
        self.batch_y = []
        self.run_done = False
        self.num_corrected_pixels = 0

    def run_model_on_tile(self, naip_tile, last_features=False, batch_size=32):
        
        with T.no_grad():
            if last_features:
                y_hat,features = self.predict_entire_image(naip_tile,last_features)
                output = y_hat[:, :, :]
                return softmax(output,2),features
            else:
                y_hat = self.predict_entire_image(naip_tile,last_features)
                output = y_hat[:, :, :]
                return softmax(output,2)

    def predict_entire_image(self, x, last_features=False):
       
        norm_image = x
        _, w, h = norm_image.shape
        
        out = np.zeros((21, w, h))

        norm_image1 = norm_image
        x_c_tensor1 = T.from_numpy(norm_image1).float().to(self.device)
        if last_features:
            y_pred1, features = self.model(x_c_tensor1.unsqueeze(0),last_features)
        else:
            y_pred1 = self.model(x_c_tensor1.unsqueeze(0))
        y_hat1 = y_pred1.cpu().numpy()
        
        out[:] = y_hat1[0,1:]
          
        pred = np.rollaxis(out, 0, 3)
        
        print(pred.shape)
        if last_features: return pred,features
        else: return pred




