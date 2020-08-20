from .ModelSessionAbstract import ModelSession
import torch as T
import numpy as np
import torch.nn as nn
import copy
import os, json
from torch.autograd import Variable
import time
from scipy.special import softmax

class CoreModel(nn.Module):
    def __init__(self):
        super(CoreModel,self).__init__()
        self.conv1 = T.nn.Conv2d(4,64,3,1,1)
        self.conv2 = T.nn.Conv2d(64,64,3,1,1)
        self.conv3 = T.nn.Conv2d(64,64,3,1,1)
        self.conv4 = T.nn.Conv2d(64,64,3,1,1)
        self.conv5 = T.nn.Conv2d(64,64,3,1,1)
       
    def forward(self,inputs):
        x = T.relu(self.conv1(inputs))
        x = T.relu(self.conv2(x))
        x = T.relu(self.conv3(x))
        x = T.relu(self.conv4(x))
        x = T.relu(self.conv5(x))
        return x

class AugmentModel(nn.Module):
    def __init__(self):
        super(AugmentModel,self).__init__()
        self.lastweight = T.nn.Parameter(T.zeros(8,64,22).uniform_())
        self.lastbias = T.nn.Parameter(T.zeros(8,22).uniform_())
        
    def forward(self,features):
        outs = (T.einsum('bfxy,mfo->bmoxy',features,self.lastweight) + self.lastbias.unsqueeze(2).unsqueeze(3)).softmax(2)
        return outs
        
class XYModel(nn.Module):
    def __init__(self):
        super(XYModel,self).__init__()
        self.pos1 = T.nn.Conv2d(2,256,1)
        self.pos5 = T.nn.Conv2d(256,8,1)

    def forward(self,inputs):
        c = self.pos1(inputs).relu()
        return self.pos5(c).softmax(1)+0.0001
    
class TorchSmoothingXYFineTune(ModelSession):

    def __init__(self, model_fn, gpuid, fine_tune_layer):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        
        self.model_fn = model_fn
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        
        self.core_model = CoreModel()
        self.augment_model = AugmentModel()
        self.pos_model = XYModel()
        
        self.init_model()

        print(sum(x.numel() for x in self.core_model.parameters()))

        for param in self.core_model.parameters():
            param.requires_grad = False
        for param in self.augment_model.parameters():
            param.requires_grad = False
        for param in self.pos_model.parameters():
            param.requires_grad = True

        self.features = None

        self.naip_data = None

        self.coord_data = None
        
        self.corr_features = []
        self.corr_coords = []
        self.corr_labels = []
        self.num_corrections_since_retrain = [ 0 ]

    @property
    def last_tile(self):
        return 0

    def run(self, naip_data, inference_mode, bounds, transform):
        print(naip_data.shape)
        print(bounds)
        x = naip_data
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        x = x[:4, :, :]
        naip_data = x / 255.0

        print(transform)

        self.last_outputs = []

        self.naip_data = naip_data  # keep non-trimmed size, i.e. with padding

        self.coord_data = np.zeros((1,2,)+naip_data.shape[1:])      
        self.coord_data[:,0] = np.random.ranf()-0.5
        self.coord_data[:,1] = np.random.ranf()-0.5

        #corner_x = bounds[0] - np.array(transform)[2]
        #corner_y = bounds[1] - np.array(transform)[5]
        corner_x = np.array(transform)[2] - 342471
        corner_y = -np.array(transform)[5] + 4311703
        print(corner_x,corner_y)

        self.coord_data[:,1] = (corner_x/5000)-0.5
        self.coord_data[:,0] = (corner_y/5000)-0.5

        self.coord_data[:,0] += np.arange(naip_data.shape[1]).reshape(-1,1)/5000
        self.coord_data[:,1] += np.arange(naip_data.shape[2])/5000

        with T.no_grad():

            if naip_data.shape[1] < 300:
                features = self.run_core_and_augment_model_on_tile(naip_data)
                self.features = features.cpu().numpy()

                weights = self.pos_model(T.from_numpy(self.coord_data).float().to(self.device))
                out = T.einsum('bmoxy,bmxy->boxy',features,weights).cpu().numpy()[0,1:]
                #out = self.features[0,0,1:]
                #out = self.features[0,1:]

                out = np.rollaxis(out, 0, 3)
                
                self.last_outputs.append(out)

            else:

                self.features, self.last_outputs = self.run_large(naip_data)

        print(self.last_outputs[0].shape)
        return self.last_outputs

    def run_large(self,naip_data):
        eval_size = 256
        batch_size = 128
        _,w,h = naip_data.shape

        features_out = np.zeros((1,64, w, h))
        preds_out = [ np.zeros((w,h,21)) for _ in range(self.num_models) ]

        x_coords, y_coords = [],[]
        x, y = 0, 0
        while x+eval_size<w:
            x_coords.append(x)
            x += eval_size-10
        x_coords.append(w-eval_size)
        while y+eval_size<h:
            y_coords.append(y)
            y += eval_size-10
        y_coords.append(h-eval_size)

        def evaluate():
            inputs = T.from_numpy(batch[:len(batch_coords)]).float().to(self.device)
            features = self.core_model(inputs)
            preds = [ model(features) for model in self.augment_models ]
            for j,c in enumerate(batch_coords):
                xj,yj = c
                features_out[0,:,xj+5:xj+eval_size-5,yj+5:yj+eval_size-5] = features[j,:,5:-5,5:-5].cpu().numpy()
                for m in range(self.num_models):
                    preds_out[m][xj+5:xj+eval_size-5,yj+5:yj+eval_size-5,:] = np.rollaxis(preds[m][j,1:,5:-5,5:-5].softmax(0).cpu().numpy(), 0, 3)

        batch = np.zeros((batch_size,4,eval_size,eval_size))
        i = 0
        batch_coords = []
        for x in x_coords:
            for y in y_coords:
                batch_coords.append((x,y))
                batch[i] = naip_data[:,x:x+eval_size,y:y+eval_size]
                i += 1
                if i == batch_size:
                    evaluate()
                    i = 0
                    batch_coords = []
        if i>0: evaluate()

        return features_out, preds_out

    def retrain(self, train_steps=1000, learning_rate=1e-2):
      
        print_every_k_steps = 409
        
        self.init_model()
        
        self.num_corrections_since_retrain.append(0)

        batch_x = T.from_numpy(np.array(self.corr_features)).float().to(self.device)
        batch_c = T.from_numpy(np.array(self.corr_coords)).float().to(self.device)
        batch_y = T.from_numpy(np.array(self.corr_labels)).to(self.device)


        if batch_x.shape[0] > 0:
        
            optimizer = T.optim.Adam(self.pos_model.parameters(), lr=learning_rate, eps=1e-5)
            criterion = T.nn.CrossEntropyLoss().to(self.device)

            for i in range(train_steps):
                #print('step %d' % i)
                acc = 0
                
                with T.enable_grad():

                    optimizer.zero_grad()
                    
                    weights = self.pos_model(batch_c.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
                    #print(weights.shape)
                    pred = T.einsum('bmc,bm->bc',batch_x,weights)

                    loss = criterion(pred,batch_y)


                    loss.backward()
                    optimizer.step()
                
                if i % print_every_k_steps == 0:
                    acc = (pred.argmax(1)==batch_y).float().mean().item()
                    print("Step pixel acc: ", acc)

                message = "Fine-tuned model with %d samples." % (len(self.corr_features))

        success = True
        print(message)
        return success, message
    
    def undo(self):
        num_undone = sum(self.num_corrections_since_retrain[-1])
        message = 'Removed {} labels'.format(' '.join(map(str,self.num_corrections_since_retrain[-1])))
        for i in range(self.num_models):
            self.corr_features[i] = self.corr_features[i][:len(self.corr_features[i])-self.num_corrections_since_retrain[-1][i]]
            self.corr_labels[i] = self.corr_labels[i][:len(self.corr_labels[i])-self.num_corrections_since_retrain[-1][i]]
            self.num_corrections_since_retrain[-1][i] = 0
        if num_undone == 0: self.num_corrections_since_retrain = self.num_corrections_since_retrain[:-1]
        if len(self.num_corrections_since_retrain) == 0:
            self.num_corrections_since_retrain = [ [ 0 for _ in range(self.num_models)] ]
        return True, message, num_undone

    def add_sample_point(self, row, col, class_idx, model_idx):
        print("adding sample: class %d (incremented to %d) at (%d, %d), model %d" % (class_idx, class_idx+1 , row, col, model_idx))

        self.corr_labels.append(class_idx+1)
        self.corr_features.append(self.features[0,:,:,row,col])
        self.corr_coords.append(self.coord_data[0,:,row,col])
        self.num_corrections_since_retrain[-1] += 1

    def init_model(self):
        checkpoint = T.load(self.model_fn, map_location=self.device)
        
        self.core_model.load_state_dict(checkpoint, strict=False)
        self.core_model.eval()
        self.core_model.to(self.device)
        
        self.augment_model.load_state_dict(checkpoint, strict=False)
        self.augment_model.eval()
        self.augment_model.to(self.device)

        self.pos_model.load_state_dict(checkpoint, strict=False)
        self.pos_model.to(self.device)
        
    def reset(self):
        self.init_model()

        self.corr_features = []
        self.corr_coords = []
        self.corr_labels = []
        self.num_corrections_since_retrain = [ 0 ]

    def run_core_and_augment_model_on_tile(self, naip_tile):
          
        _, w, h = naip_tile.shape

        x_c_tensor1 = T.from_numpy(naip_tile).float().to(self.device)
        features = self.augment_model(self.core_model(x_c_tensor1.unsqueeze(0)))
           
        return features

    def save_state_to(self, directory):
        pass

    def load_state_from(self, directory):
        pass

    def record_cycle(self, model_idx):
        print("Cycling to: %d" % (model_idx))

        return "test"