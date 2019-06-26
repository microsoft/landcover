import numpy as np
import keras.utils

# def data_generator_sr(fns, batch_size, input_size, output_size, num_channels, superres_states=[], verbose=None):
#     file_indices = list(range(len(fns)))
    
#     x_batch = np.zeros((batch_size, input_size, input_size, num_channels), dtype=np.float32)
#     y_hr_batch = np.zeros((batch_size, output_size, output_size, 5), dtype=np.float32)
#     y_sr_batch = np.zeros((batch_size, output_size, output_size, 22), dtype=np.float32)
    
#     counter = 0
#     while 1:
#         np.random.shuffle(file_indices)
        
#         for i in range(0, len(file_indices), batch_size):

#             if i + batch_size >= len(file_indices): # if we don't have enough samples left, just quit and reshuffle
#                 break

#             batch_idx = 0
#             for j in range(i, i+batch_size):
#                 fn = str(fns[file_indices[j]])
#                 fn_parts = fn.split("/")

#                 data = np.load().squeeze()
#                 data = np.rollaxis(data, 0, 3)

#                 y_train_hr = data[:,:,4]
#                 y_train_hr[y_train_hr==15] = 0
#                 y_train_hr[y_train_hr==5] = 4
#                 y_train_hr[y_train_hr==6] = 4
#                 y_train_hr = keras.utils.to_categorical(y_train_hr, 5)
                
#                 y_train_nlcd = data[:,:,5]
#                 y_train_nlcd = keras.utils.to_categorical(y_train_nlcd, 22)
#                 if fn_parts[5] in superres_states:
#                     y_train_hr[:,:,0] = 0
#                 else:
#                     y_train_hr[:,:,0] = 1

#                 x_batch[batch_idx] = data[:,:,:4]
#                 y_hr_batch[batch_idx] = y_train_hr
#                 y_sr_batch[batch_idx] = y_train_nlcd
#                 batch_idx += 1

#             yield (x_batch.copy(), {"outputs_hr": y_hr_batch.copy(), "outputs_sr": y_sr_batch.copy()})
#             if verbose is not None:
#                 print("%s yielded %d" % (verbose, counter))
#             counter += 1

# def data_generator_hr(fns, batch_size, input_size, output_size, num_channels, verbose=None):
#     file_indices = list(range(len(fns)))
    
#     x_batch = np.zeros((batch_size, input_size, input_size, num_channels), dtype=np.float32)
#     y_batch = np.zeros((batch_size, output_size, output_size, 5), dtype=np.float32)
    
#     counter = 0
#     while 1:
#         np.random.shuffle(file_indices)
        
#         for i in range(0, len(file_indices), batch_size):

#             if i + batch_size >= len(file_indices): # if we don't have enough samples left, just quit and reshuffle
#                 break

#             batch_idx = 0
#             for j in range(i, i+batch_size):
#                 fn = fns[file_indices[j]]

#                 data = np.load(fn).squeeze()
#                 data = np.rollaxis(data, 0, 3)

#                 x_batch[batch_idx] = data[:,:,:4]
                
#                 y_train_hr = data[:,:,4]
#                 y_train_hr[y_train_hr==15] = 0
#                 y_train_hr[y_train_hr==5] = 4
#                 y_train_hr[y_train_hr==6] = 4
#                 y_train_hr = keras.utils.to_categorical(y_train_hr, 5)
#                 #y_train_hr[:,:,0] = 1
#                 y_batch[batch_idx] = y_train_hr

#                 batch_idx += 1

#             yield (x_batch.copy(), y_batch.copy())
#             if verbose is not None:
#                 print("%s yielded %d" % (verbose, counter))
#             counter += 1

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    #def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1, n_classes=10, shuffle=True):
    def __init__(self, patches, batch_size, steps_per_epoch, input_size, output_size, num_channels, superres=False, superres_states=[]):
        'Initialization'

        self.patches = patches
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        assert steps_per_epoch * batch_size < len(patches)

        self.input_size = input_size
        self.output_size = output_size

        self.num_channels = num_channels
        self.num_classes = 5

        self.superres = superres
        self.superres_states = superres_states
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        fns = [self.patches[i] for i in indices]

        x_batch = np.zeros((self.batch_size, self.input_size, self.input_size, self.num_channels), dtype=np.float32)
        y_hr_batch = np.zeros((self.batch_size, self.output_size, self.output_size, 5), dtype=np.float32)
        y_sr_batch = None
        if self.superres:
            y_sr_batch = np.zeros((self.batch_size, self.output_size, self.output_size, 22), dtype=np.float32)
        
        for i, fn in enumerate(fns):
            fn_parts = fn.split("/")

            data = np.load(fn).squeeze()
            data = np.rollaxis(data, 0, 3)
            
            # setup x
            x_batch[i] = data[:,:,:4]

            # setup y_highres
            y_train_hr = data[:,:,4]
            y_train_hr[y_train_hr==15] = 0
            y_train_hr[y_train_hr==5] = 4
            y_train_hr[y_train_hr==6] = 4
            y_train_hr = keras.utils.to_categorical(y_train_hr, 5)

            if self.superres:
                if fn_parts[5] in self.superres_states:
                    y_train_hr[:,:,0] = 0
                else:
                    y_train_hr[:,:,0] = 1
            else:
                y_train_hr[:,:,0] = 0
            y_hr_batch[i] = y_train_hr
            
            # setup y_superres
            if self.superres:
                y_train_nlcd = data[:,:,5]
                y_train_nlcd = keras.utils.to_categorical(y_train_nlcd, 22)
                y_sr_batch[i] = y_train_nlcd

        
        if self.superres:
            return x_batch.copy(), {"outputs_hr": y_hr_batch, "outputs_sr": y_sr_batch}
        else:
            return x_batch.copy(), y_hr_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.patches))
        np.random.shuffle(self.indices)