import sys, os, time, copy

import numpy as np

import sklearn.base
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras import optimizers

from ServerModelsAbstract import BackendModel

from web_tool import ROOT_DIR

AUGMENT_MODEL = MLPClassifier(
    hidden_layer_sizes=(),
    activation='relu',
    alpha=0.0001,
    solver='lbfgs',
    tol=0.0001,
    verbose=False,
    validation_fraction=0.0,
    n_iter_no_change=10
)


class KerasDenseFineTune(BackendModel):

    def __init__(self, model_fn, gpuid, superres=False, verbose=False):
        # Load model
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        import keras
        import keras.models
        import keras.backend as K

        self.model_fn = model_fn
        
        tmodel = keras.models.load_model(self.model_fn, compile=False, custom_objects={
            "jaccard_loss":keras.metrics.mean_squared_error, 
            "loss":keras.metrics.mean_squared_error
        })

        #with open("web_tool/data/final_model.json", 'r') as json_file:
        #    tmodel = keras.models.model_from_json(json_file.read())
        #tmodel.load_weights("web_tool/data/final_model_weights.h5")

        feature_layer_idx = None
        if superres:
            feature_layer_idx = -4
        else:
            feature_layer_idx = -3
        
        self.model = keras.models.Model(inputs=tmodel.inputs, outputs=[tmodel.outputs[0], tmodel.layers[feature_layer_idx].output])
        self.model.compile("sgd","mse")
        self.model._make_predict_function()

        self.output_channels = self.model.output_shape[0][3]
        self.output_features = self.model.output_shape[1][3]
        self.input_size = self.model.input_shape[1]

        self.down_weight_padding = 40

        self.stride_x = self.input_size - self.down_weight_padding*2
        self.stride_y = self.input_size - self.down_weight_padding*2

        self.verbose = verbose

        # Seed augmentation model dataset
        self.current_features = None

        self.augment_base_x_train = []
        self.augment_base_y_train = []

        self.augment_x_train = []
        self.augment_y_train = []
        self.augment_model = sklearn.base.clone(AUGMENT_MODEL)
        self.augment_model_trained = False

        self.undo_stack = []

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
        

    def run(self, naip_data, extent, on_tile=False):
        ''' Expects naip_data to have shape (height, width, channels) and have values in the [0, 255] range.
        '''
        naip_data = naip_data / 255.0
        output, output_features = self.run_model_on_tile(naip_data)
        
        if self.augment_model_trained:
            original_shape = output.shape
            output = output_features.reshape(-1, output_features.shape[2])
            output = self.augment_model.predict_proba(output)
            output = output.reshape(original_shape[0], original_shape[1],  -1)

        if not on_tile:
            self.current_features = output_features

        return output

    def run_model_on_batch(self, batch_data, batch_size=32, predict_central_pixel_only=False):
        ''' Expects batch_data to have shape (none, 240, 240, 4) and have values in the [0, 255] range.
        '''
        batch_data = batch_data / 255.0
        output = self.model.predict(batch_data, batch_size=batch_size, verbose=0)
        output, output_features = output
        output = output[:,:,:,1:]

        if self.augment_model_trained:
            num_samples, height, width, num_features = output_features.shape

            if predict_central_pixel_only:
                output_features = output_features[:,120,120,:].reshape(-1, num_features)
                output = self.augment_model.predict_proba(output_features)
                output = output.reshape(num_samples, 4)
            else:
                output_features = output_features.reshape(-1, num_features)
                output = self.augment_model.predict_proba(output_features)
                output = output.reshape(num_samples, height, width, 4)
        else:
            if predict_central_pixel_only:
                output = output[:,120,120,:]
        
        return output

    def retrain(self, **kwargs):
        x_train = np.concatenate(self.augment_x_train, axis=0)
        y_train = np.concatenate(self.augment_y_train, axis=0)
        
        vals, counts = np.unique(y_train, return_counts=True)

        if len(vals) >= 4:
            self.augment_model.fit(x_train, y_train)
            self.augment_model_trained = True
            self.undo_stack.append("retrain")

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
        self.undo_stack.append("sample")

    def undo(self):
        num_undone = 0
        if len(self.undo_stack) > 0:
            undo = self.undo_stack.pop()
            if undo == "sample":
                self.augment_x_train.pop()
                self.augment_y_train.pop()
                num_undone += 1
                success = True
                message = "Undoing sample"
            elif undo == "retrain":
                while self.undo_stack[-1] == "retrain":
                    self.undo_stack.pop()
                self.augment_x_train.pop()
                self.augment_y_train.pop()
                num_undone += 1
                success = True
                message = "Undoing sample"
            else:
                raise ValueError("This shouldn't happen")
        else:
            success = False
            message = "Nothing to undo"
        return success, message, num_undone

    def reset(self):
        self.augment_x_train = []
        self.augment_y_train = []
        self.undo_stack = []
        self.augment_model = sklearn.base.clone(AUGMENT_MODEL)
        self.augment_model_trained = False

        for row in self.augment_base_x_train:
            self.augment_x_train.append(row)
        for row in self.augment_base_y_train:
            self.augment_y_train.append(row)

    def run_model_on_tile(self, naip_tile, batch_size=32):
        ''' Expects naip_tile to have shape (height, width, channels) and have values in the [0, 1] range.
        '''
        height = naip_tile.shape[0]
        width = naip_tile.shape[1]
        
        output = np.zeros((height, width, self.output_channels), dtype=np.float32)
        output_features = np.zeros((height, width, self.output_features), dtype=np.float32)

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
            output[y:y+self.input_size, x:x+self.input_size] += model_output[0][i] * kernel[..., np.newaxis]
            output_features[y:y+self.input_size, x:x+self.input_size] += model_output[1][i] * kernel[..., np.newaxis]
            counts[y:y+self.input_size, x:x+self.input_size] += kernel

        output = output / counts[..., np.newaxis]
        output = output[:,:,1:5]
        output_features = output_features / counts[..., np.newaxis]

        return output, output_features



class KerasBackPropFineTune(BackendModel):

    def __init__(self, model_fn, gpuid, superres=False, verbose=False):

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
        
        self.num_output_channels = self.model.output_shape[3]
        self.input_size = self.model.input_shape[1]        

        self.naip_data = None
        self.correction_labels = None

        self.down_weight_padding = 40

        self.stride_x = self.input_size - self.down_weight_padding*2
        self.stride_y = self.input_size - self.down_weight_padding*2

        self.batch_x = []
        self.batch_y = []
        self.num_corrected_pixels = 0
        
        self.verbose = verbose
        
    def run(self, naip_data, extent, on_tile=False):
        ''' Expects naip_data to have shape (height, width, channels) and have values in the [0, 255] range.
        '''
        # If we click somewhere else before retraining we need to commit the current set of training samples
        if self.correction_labels is not None and not on_tile:
            self.process_correction_labels()

        naip_data = naip_data / 255.0
        height = naip_data.shape[0]
        width = naip_data.shape[1]
        output = self.run_model_on_tile(naip_data)

        # Reset the state of our retraining mechanism
        if not on_tile:
            self.correction_labels = np.zeros((height, width, self.num_output_channels), dtype=np.float32)
            self.naip_data = naip_data.copy()
        
        return output

    def run_model_on_batch(self, batch_data, batch_size=32, predict_central_pixel_only=False):
        ''' Expects batch_data to have shape (none, 240, 240, 4) and have values in the [0, 255] range.
        '''
        output = output / 255.0
        output = self.model.predict(batch_data, batch_size=batch_size, verbose=0)
        output = output[:,:,:,1:]

        if predict_central_pixel_only:
            output = output[:,120,120,:]
        
        return output

    def retrain(self, number_of_steps=5, last_k_layers=3, learning_rate=0.01, batch_size=32, **kwargs):
        # Commit any training samples we have received to the training set
        self.process_correction_labels()

        # Reset the model to the initial state
        num_layers = len(self.model.layers)
        for i in range(num_layers):
            if self.model.layers[i].trainable:
                self.model.layers[i].set_weights(self.old_model.layers[i].get_weights())
            self.model.layers[i].trainable = False

        for i in range(num_layers-last_k_layers, num_layers):
            self.model.layers[i].trainable = True
        self.model.compile(optimizers.Adam(lr=learning_rate, amsgrad=True), "categorical_crossentropy")

        if len(self.batch_x) > 0:

            x_train = np.array(self.batch_x)
            y_train = np.array(self.batch_y)
            y_train_labels = y_train.argmax(axis=3)

            # Perform retraining
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
                history.append(np.mean(training_losses))
            beginning_loss = history[0]
            end_loss = history[-1]
            
            # Evaluate training accuracy - surrogate for how well we are able to fit our supplemental training set
            y_pred = self.model.predict(x_train)        
            y_pred_labels = y_pred.argmax(axis=3)
            mask = y_train_labels != 0
            acc = np.sum(y_train_labels[mask] == y_pred_labels[mask]) / np.sum(mask)
            
            # The front end expects some return message
            success = True
            message = "Re-trained model with %d samples<br>Starting loss:%f<br>Ending loss:%f<br>Training acc: %f." % (
                x_train.shape[0],
                beginning_loss, end_loss,
                acc
            )
        else:
            success = False
            message = "Need to add labels before you can retrain"

        return success, message

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):        
        self.correction_labels[tdst_row:bdst_row+1, tdst_col:bdst_col+1, class_idx+1] = 1.0

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
        self.batch_x.extend(batch_x)
        self.batch_y.extend(batch_y)

    def undo(self):
        return False, "Not implemented yet"

    def reset(self):

        self.model = copy.deepcopy(self.old_model)
        self.batch_x = []
        self.batch_y = []
        self.naip_data = None
        self.correction_labels = None
        
    def run_model_on_tile(self, naip_tile, batch_size=32):
        ''' Expects naip_tile to have shape (height, width, channels) and have values in the [0, 1] range.
        '''
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
    
