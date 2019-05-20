import sys, os, time, copy

import numpy as np
import pdb

import sklearn.base
from sklearn.neural_network import MLPClassifier
from keras import optimizers

from ServerModelsAbstract import BackendModel

from web_tool.frontend_server import ROOT_DIR


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
        import keras.backend as K

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
            seed_x_fn = ROOT_DIR + "/data/seed_data_hr+sr_x.npy"
            seed_y_fn = ROOT_DIR + "/data/seed_data_hr+sr_y.npy"
        else:
            seed_x_fn = ROOT_DIR + "/data/seed_data_hr_x.npy"
            seed_y_fn = ROOT_DIR + "/data/seed_data_hr_y.npy"
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
            output = output.reshape(original_shape[0], original_shape[1],  -1)

        # apply padding to the output_features
        if padding > 0:
            output_features = output_features[padding:-padding,padding:-padding,:]
        self.current_features = output_features

        return output

    def retrain(self, **kwargs):
        x_train = np.concatenate(self.augment_x_train, axis=0)
        y_train = np.concatenate(self.augment_y_train, axis=0)
        
        vals, counts = np.unique(y_train, return_counts=True)

        if len(vals) >= 4:
            print("Fitting model with %d samples of %d different classes" % (x_train.shape[0], len(vals)))
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


class KerasBackPropFineTune(BackendModel):

    def __init__(self, model_fn, gpuid, superres=False):

        # ------------------------------------------------------
        # Step 1
        #   Load Keras model
        # ------------------------------------------------------
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        import keras
        import keras.models
        import keras.backend as K
        K.set_learning_phase(0)

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
        
        self.model = keras.models.Model(inputs=tmodel.inputs, outputs=tmodel.outputs[0])
        for layer in self.model.layers:
            layer.trainable = False
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer._per_input_updates = {}
                layer.training = False
        self.model.compile("sgd","categorical_crossentropy")
        
        self.old_model = copy.deepcopy(self.model)
        
        self.output_channels = self.model.output_shape[3]
        self.input_size = self.model.input_shape[1]        

        self.naip_data = None
        self.correction_labels = None
        self.tile_padding = 0

        self.down_weight_padding = 40

        self.stride_x = self.input_size - self.down_weight_padding*2
        self.stride_y = self.input_size - self.down_weight_padding*2

        self.batch_x = []
        self.batch_y = []
        self.num_corrected_pixels = 0
        
        # pdb.set_trace()
        
    def run(self, naip_data, naip_fn, extent, padding):

        if self.correction_labels is not None:
            self.process_correction_labels()

        naip_data = naip_data / 255.0
        height = naip_data.shape[0]
        width = naip_data.shape[1]
        output = self.run_model_on_tile(naip_data)

        self.tile_padding = padding
        self.correction_labels = np.zeros((height, width, self.output_channels), dtype=np.float32)
        self.naip_data = naip_data.copy()
        self.last_output = output
        
        return output

    def retrain(self, number_of_steps=8, last_k_layers=3, learning_rate=0.015, batch_size=32, **kwargs):
        
        self.process_correction_labels()

        num_layers = len(self.model.layers)
        for i in range(num_layers):
            if self.model.layers[i].trainable:
                print("Reseting layer %d" % (i))
                self.model.layers[i].set_weights(self.old_model.layers[i].get_weights())
            self.model.layers[i].trainable = False

        for i in range(num_layers-last_k_layers, num_layers):
            self.model.layers[i].trainable = True

        self.model.compile(optimizers.Adam(lr=learning_rate, amsgrad=True), "categorical_crossentropy")
        self.model.summary()

        if len(self.batch_x) > 1:

            x_train = np.array(self.batch_x)
            y_train = np.array(self.batch_y)
            print("Training set shape: ", x_train.shape, y_train.shape)
            y_train_labels = y_train.argmax(axis=3)
            print("Label set: ", np.unique(y_train_labels[y_train_labels!=0], return_counts=True))
            print("Starting fine-tuning for %d steps over the last %d layers using %d samples with lr of %f" % 
                (number_of_steps, last_k_layers, len(self.batch_x), learning_rate)
            )

            '''
            history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=number_of_steps
            )
            '''
            
            history = []
            for i in range(number_of_steps):
                idxs = np.arange(x_train.shape[0])
                np.random.shuffle(idxs)
                x_train = x_train[idxs]
                y_train = y_train[idxs]
                
                training_losses = []
                for j in range(0, x_train.shape[0], batch_size):
                    batch_x = x_train[j:j+batch_size]
                    batch_y = y_train[j:j+batch_size]

                    actual_batch_size = batch_x.shape[0]

                    training_loss = self.model.train_on_batch(batch_x, batch_y)
                    training_losses.append(training_loss)
                print(i, np.mean(training_losses))
                history.append(np.mean(training_losses))
            
            '''
            beginning_loss = history.history["loss"][0]
            end_loss = history.history["loss"][-1]
            '''
            beginning_loss = history[0]
            end_loss = history[-1]
            
            
            y_pred = self.model.predict(x_train)        
            y_pred_labels = y_pred.argmax(axis=3)
            mask = y_train_labels != 0
            acc = np.sum(y_train_labels[mask] == y_pred_labels[mask]) / np.sum(mask)
            print("training acc", acc)
            print("training loss", beginning_loss, end_loss)
            
            
            success = True
            message = "Re-trained model with %d samples<br>Starting loss:%f<br>Ending loss:%f<br>Training acc: %f." % (
                x_train.shape[0],
                beginning_loss, end_loss,
                acc
            )
            message = "Fit accessory model with %d samples" % (x_train.shape[0])
            
            return success, message
        else:
            return False, "Need to add labels"

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        
        padding = self.tile_padding
        height = self.naip_data.shape[0]
        width = self.naip_data.shape[1]

        self.correction_labels[tdst_row + padding : bdst_row + 1 + padding, tdst_col + padding : bdst_col + 1 + padding, class_idx + 1] = 1.0

        print("Adding correction using class id %d" % (class_idx))
        print("Correction shape: ", self.correction_labels[tdst_row + padding : bdst_row + 1 + padding, tdst_col + padding : bdst_col + 1 + padding, :].shape)


    def process_correction_labels(self):

        height = self.naip_data.shape[0]
        width = self.naip_data.shape[1]

        batch_x = []
        batch_y = []
        batch_count = 0
        num_skips = 0
        for y_index in (list(range(0, height - self.input_size, self.stride_y)) + [height - self.input_size,]):
            for x_index in (list(range(0, width - self.input_size, self.stride_x)) + [width - self.input_size,]):
                naip_im = self.naip_data[y_index:y_index+self.input_size, x_index:x_index+self.input_size, :].copy()
                correction_labels_slice = self.correction_labels[y_index:y_index+self.input_size, x_index:x_index+self.input_size, :].copy()

                if not np.all(correction_labels_slice == 0):
                    batch_x.append(naip_im)
                    batch_y.append(correction_labels_slice)
                    self.correction_labels[y_index:y_index+self.input_size, x_index:x_index+self.input_size, :] = 0
                else:
                    num_skips += 1
        print("Added %d samples, skipped %d samples" % (len(batch_x), num_skips))
        self.batch_x.extend(batch_x)
        self.batch_y.extend(batch_y)

        
    def reset(self):
        self.model = copy.deepcopy(self.old_model)
        self.batch_x = []
        self.batch_y = []
        self.naip_data = None
        self.tile_padding = 0
        self.correction_labels = None
        
    def run_model_on_tile(self, naip_tile, batch_size=32):
        height = naip_tile.shape[0]
        width = naip_tile.shape[1]
        
        output = np.zeros((height, width, self.output_channels), dtype=np.float32)
        
        counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
        kernel = np.ones((self.input_size, self.input_size), dtype=np.float32) * 0.1
        kernel[10:-10, 10:-10] = 1
        kernel[self.down_weight_padding:self.down_weight_padding+self.stride_y,
               self.down_weight_padding:self.down_weight_padding+self.stride_x] = 5

        batch = []
        batch_indices = []
        batch_count = 0

        for y_index in (list(range(0, height - self.input_size, self.stride_y)) + [height - self.input_size,]):
            for x_index in (list(range(0, width - self.input_size, self.stride_x)) + [width - self.input_size,]):
                naip_im = naip_tile[y_index:y_index+self.input_size, x_index:x_index+self.input_size, :]

                batch.append(naip_im)
                batch_indices.append((y_index, x_index))
                batch_count+=1


        model_output = self.model.predict(np.array(batch), batch_size=batch_size, verbose=0)
        
        for i, (y, x) in enumerate(batch_indices):
            output[y:y+self.input_size, x:x+self.input_size] += model_output[i] * kernel[..., np.newaxis]
            counts[y:y+self.input_size, x:x+self.input_size] += kernel

        output = output / counts[..., np.newaxis]
        output = output[:,:,1:5]

        return output
    
