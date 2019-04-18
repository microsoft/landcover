import sys, os, time

import numpy as np

import sklearn.base
from sklearn.neural_network import MLPClassifier

from ServerModelsAbstract import BackendModel

AUGMENT_MODEL = MLPClassifier(
    hidden_layer_sizes=(),
    activation='relu',
    alpha=0.001,
    solver='lbfgs',
    verbose=False,
    validation_fraction=0.0,
    n_iter_no_change=10
)

class KerasDenseFineTune(BackendModel):

    def __init__(self, model_fn, gpuid, superres=False):

        # ------------------------------------------------------
        # Step 1
        #   Load Keras model
        # ------------------------------------------------------
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        import keras
        import keras.models

        self.model_fn = model_fn
        
        tmodel = keras.models.load_model(self.model_fn, custom_objects={
            "jaccard_loss":keras.metrics.mean_squared_error, 
            "loss":keras.metrics.mean_squared_error
        })

        feature_layer_idx = None
        if superres:
            feature_layer_idx = -4
        else:
            feature_layer_idx = -3
        
        self.model = keras.models.Model(inputs=tmodel.inputs, outputs=[tmodel.outputs[0], tmodel.layers[feature_layer_idx].output])
        self.model.compile("sgd","mse")

        self.output_channels = self.model.output_shape[0][3]
        self.output_features = self.model.output_shape[1][3]
        self.input_size = self.model.input_shape[1]

        # ------------------------------------------------------
        # Step 2
        #   Pre-load augment model seed data
        # ------------------------------------------------------
        self.current_features = None

        self.augment_base_x_train = []
        self.augment_base_y_train = []

        self.augment_x_train = []
        self.augment_y_train = []
        self.augment_model = sklearn.base.clone(AUGMENT_MODEL)
        self.augment_model_trained = False

        seed_x_fn = ""
        seed_y_fn = ""
        if superres:
            seed_x_fn = "web-tool/data/seed_data_hr+sr_x.npy"
            seed_y_fn = "web-tool/data/seed_data_hr+sr_y.npy"
        else:
            seed_x_fn = "web-tool/data/seed_data_hr_x.npy"
            seed_y_fn = "web-tool/data/seed_data_hr_y.npy"
        for row in np.load(seed_x_fn):
            self.augment_base_x_train.append(row)
        for row in np.load(seed_y_fn):
            self.augment_base_y_train.append(row)

        for row in self.augment_base_x_train:
            self.augment_x_train.append(row)
        for row in self.augment_base_y_train:
            self.augment_y_train.append(row)
        

    def run(self, naip_data, naip_fn, extent, padding):
        output, output_features = self.run_model_on_tile(naip_data)
        
        if self.augment_model_trained:
            original_shape = output.shape
            output = output_features.reshape(-1, output_features.shape[2])
            output = self.augment_model.predict_proba(output)
            output = output.reshape(original_shape)

        # apply padding to the output_features
        if padding > 0:
            output_features = output_features[padding:-padding,padding:-padding,:]
        self.current_features = output_features

        return output

    def retrain(self):
        x_train = np.concatenate(self.augment_x_train, axis=0)
        y_train = np.concatenate(self.augment_y_train, axis=0)
        
        vals, counts = np.unique(y_train, return_counts=True)

        if len(vals) == 4:
            print("Fitting model with %d samples" % (x_train.shape[0]))
            self.augment_model.fit(x_train, y_train)
            self.augment_model_trained = True

            success = True
            message = "Fit accessory model with %d samples" % (x_train.shape[0])
        else:
            success = False
            message = "Need to include training samples from each class"
        
        return success, message

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        x_samples = self.current_features[tdst_row:bdst_row+1, tdst_col:bdst_col+1, :].copy().reshape(-1, self.current_features.shape[2])
        y_samples = np.zeros((x_samples.shape[0]), dtype=np.uint8)
        y_samples[:] = class_idx
        self.augment_x_train.append(x_samples)
        self.augment_y_train.append(y_samples)

    def reset(self):
        self.augment_x_train = []
        self.augment_y_train = []
        self.augment_model = sklearn.base.clone(AUGMENT_MODEL)
        self.augment_model_trained = False

        for row in self.augment_base_x_train:
            self.augment_x_train.append(row)
        for row in self.augment_base_y_train:
            self.augment_y_train.append(row)

    def run_model_on_tile(self, naip_tile, batch_size=32):

        naip_tile = naip_tile / 255.0

        down_weight_padding = 40
        height = naip_tile.shape[0]
        width = naip_tile.shape[1]

        stride_x = self.input_size - down_weight_padding*2
        stride_y = self.input_size - down_weight_padding*2

        output = np.zeros((height, width, self.output_channels), dtype=np.float32)
        output_features = np.zeros((height, width, self.output_features), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
        kernel = np.ones((self.input_size, self.input_size), dtype=np.float32) * 0.1
        kernel[10:-10, 10:-10] = 1
        kernel[down_weight_padding:down_weight_padding+stride_y,
            down_weight_padding:down_weight_padding+stride_x] = 5

        batch = []
        batch_indices = []
        batch_count = 0

        for y_index in (list(range(0, height - self.input_size, stride_y)) + [height - self.input_size,]):
            for x_index in (list(range(0, width - self.input_size, stride_x)) + [width - self.input_size,]):
                naip_im = naip_tile[y_index:y_index+self.input_size, x_index:x_index+self.input_size, :]

                batch.append(naip_im)
                batch_indices.append((y_index, x_index))
                batch_count+=1

        model_output = self.model.predict(np.array(batch), batch_size=batch_size, verbose=0)
        
        for i, (y, x) in enumerate(batch_indices):
            output[y:y+self.input_size, x:x+self.input_size] += model_output[0][i] * kernel[..., np.newaxis]
            output_features[y:y+self.input_size, x:x+self.input_size] += model_output[1][i] * kernel[..., np.newaxis]
            counts[y:y+self.input_size, x:x+self.input_size] += kernel

        output = output / counts[..., np.newaxis]
        output = output[:,:,1:5]

        output_features = output_features / counts[..., np.newaxis]

        return output, output_features
