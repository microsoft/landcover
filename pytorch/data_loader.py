import torch
from torch.utils import data
import numpy as np

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class DataGenerator(data.Dataset):
    'Generates data for pytorch'
    # def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1, n_classes=10, shuffle=True):
    def __init__(self, patches, batch_size, patch_size, num_channels, transform = None, superres=False,
                 superres_states=[]):
        'Initialization'
        if not transform:
            if superres:
                transform = lambda x, y_hr_batch, y_sr_batch: (x, y_hr_batch, y_sr_batch)
            else:
                transform = lambda x, y_hr_batch: (x, y_hr_batch)
        self.patches = patches
        self.batch_size = batch_size
        self.transform = transform
        self.patch_size = patch_size

        self.num_channels = num_channels
        self.num_classes = 5

        self.superres = superres
        self.superres_states = superres_states
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.patches)

    def __getitem__(self, index):
        'Generate one sample of data'
        #indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        #fns = [self.patches[i] for i in indices]

        fn_parts = self.patches[index].split("/")

        data = np.load(self.patches[index]).squeeze()
        data = np.rollaxis(data, 0, 3)

            # setup x
            #x_batch[i] = data[:, :, :4]
        x = np.transpose(data[:, :, :4], (2, 0, 1))

            # setup y_highres
        y_train_hr = data[:, :, 4]
        y_train_hr[y_train_hr == 15] = 0
        y_train_hr[y_train_hr == 5] = 4
        y_train_hr[y_train_hr == 6] = 4
        #y_train_hr = to_categorical(y_train_hr.astype(np.uint8), 5)

       # if self.superres:
        #    if fn_parts[5] in self.superres_states:
        #        y_train_hr[:, :, 0] = 0
         #   else:
        #        y_train_hr[:, :, 0] = 1
      #  else:
         #   y_train_hr[:, :, 0] = 0

            # setup y_superres
        if self.superres:
            y_train_nlcd = data[:, :, 5].astype(np.uint8)
           # y_train_nlcd = to_categorical(y_train_nlcd, 22)

        if self.superres:
            return self.transform(torch.from_numpy(x.copy()), torch.from_numpy(np.expand_dims(y_train_hr,0)), torch.from_numpy(np.expand_dims(y_train_nlcd,0)))
        else:
            return self.transform(torch.from_numpy(x.copy()), torch.from_numpy(np.expand_dims(y_train_hr,0)))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.patches))
        np.random.shuffle(self.indices)

