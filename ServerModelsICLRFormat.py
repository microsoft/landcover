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
        self.model = keras.models.load_model(self.model_fn, custom_objects={"jaccard_loss":keras.metrics.mean_squared_error})
        
        self.output_channels = self.model.output_shape[3]
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
            output[y:y+self.input_size, x:x+self.input_size] += model_output[i] * kernel[..., np.newaxis]
            counts[y:y+self.input_size, x:x+self.input_size] += kernel

        output = output / counts[..., np.newaxis]
        output[:,:,4] += output[:,:,5]
        output[:,:,4] += output[:,:,6]
        output = output[:,:,1:5]
        return output



class CNTKModel(BackendModel):

    def __init__(self, model_fn, gpuid):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        import cntk

        cntk.try_set_default_device(cntk.gpu(0))
        cntk.use_default_device()

        self.model_fn = model_fn
        self.model = cntk.load_model(self.model_fn)
        
        #self.output_channels = self.model.output_shape[3]
        #self.input_size = self.model.input_shape[1]

    def run(self, naip_data, naip_fn, extent, buffer):
        return self.run_model_on_tile(naip_data), os.path.basename(self.model_fn)

    def run_model_on_tile(self, naip, batch_size=32):

        naip = naip/255.0
        naip = naip.astype(np.float32)
        #landsat = landsat/65536.0

        naip = np.swapaxes(naip, 0, 1)
        naip = np.swapaxes(naip, 0, 2)
        #landsat = np.swapaxes(landsat, 0, 1)
        #landsat = np.swapaxes(landsat, 0, 2)
        #blg = np.swapaxes(blg, 0, 1)
        #blg = np.swapaxes(blg, 0, 2)
        x1, x2, x3 = naip.shape
        #y1, y2, y3 = landsat.shape
        #z1, z2, z3 = blg.shape
        #s2 = min(x2, y2, z2)
        #s3 = min(x3, y3, z3)
        s2 = x2
        s3 = x3

        #tile = np.concatenate((naip[:, :s2, :s3], landsat[:, :s2, :s3], blg[:, :s2, :s3]),
        #        axis=0).astype(np.float32)
        #tile = tile[:, 21:-21, 21:-21]

        tile = naip[:, :s2, :s3]

        height = tile.shape[1]
        width = tile.shape[2]
        (_, model_width, model_height) = self.model.arguments[0].shape
        focus_rad = 32

        stride_x = model_width - focus_rad*2
        stride_y = model_height - focus_rad*2

        output = np.zeros((height, width, 5), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32) + 0.0001
        kernel = np.ones((model_height, model_width), dtype=np.float32) * 0.01
        kernel[10:-10, 10:-10] = 1
        kernel[focus_rad:focus_rad+stride_y, focus_rad:focus_rad+stride_x] = 5

        batch = []
        batch_indices = []
        batch_count = 0

        for y_index in (list(range(0, height - model_height, stride_y)) + [height - model_height,]):
            for x_index in (list(range(0, width - model_width, stride_x)) + [width - model_width,]):
                batch_indices.append((y_index, x_index))
                batch.append(tile[:, y_index:y_index+model_height, x_index:x_index+model_width])
                batch_count+=1

                if batch_count == batch_size:
                    # run batch
                    model_output = self.model.eval(np.array(batch))
                    model_output = np.swapaxes(model_output,1,3)
                    model_output = np.swapaxes(model_output,1,2)
                    model_output = softmax(model_output)

                    for i, (y, x) in enumerate(batch_indices):
                        output[y:y+model_height, x:x+model_width] += model_output[i] * kernel[..., np.newaxis]
                        counts[y:y+model_height, x:x+model_width] += kernel

                    # reset batch
                    batch = []
                    batch_indices = []
                    batch_count = 0

        if batch_count > 0:
            model_output = self.model.eval(np.array(batch))
            model_output = np.swapaxes(model_output,1,3)
            model_output = np.swapaxes(model_output,1,2)
            model_output = softmax(model_output)
            for i, (y, x) in enumerate(batch_indices):
                output[y:y+model_height, x:x+model_width] += model_output[i] * kernel[..., np.newaxis]
                counts[y:y+model_height, x:x+model_width] += kernel

        out = output/counts[..., np.newaxis]
        #out[:, :, 4] += out[:, :, 5] + out[:, :, 6]
        #out[:, :, 5:] = 0

        return out[..., 1:]