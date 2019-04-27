import sys, os, time

import numpy as np

from ServerModelsAbstract import BackendModel

def softmax(output):
    output_max = np.max(output, axis=3, keepdims=True)
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=3, keepdims=True)
    return exps/exp_sums

class KerasModel(BackendModel):

    def __init__(self, model_fn, gpuid):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        import keras
        import keras.models

        self.model_fn = model_fn
        
        tmodel = keras.models.load_model(self.model_fn, custom_objects={"jaccard_loss":keras.metrics.mean_squared_error})
        self.model = keras.models.Model(inputs=tmodel.inputs, outputs=[tmodel.output, tmodel.layers[-2].output])
        self.model.compile("sgd","mse")

        self.output_channels = self.model.output_shape[0][3]
        self.output_features = self.model.output_shape[1][3]
        self.input_size = self.model.input_shape[1]

    def run(self, naip_data, naip_fn, extent, buffer):
        return self.run_model_on_tile(naip_data), os.path.basename(self.model_fn)

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
        output[:,:,4] += output[:,:,5]
        output[:,:,4] += output[:,:,6]
        output = output[:,:,1:5]

        output_features = output_features / counts[..., np.newaxis]

        return output, output_features