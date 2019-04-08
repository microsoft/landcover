import sys
import os
import time
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import cntk
import cntk.io
import string

import random
import numpy as np
import shapely
import shapely.geometry
import rasterio
import rasterio.mask
import glob
import scipy.ndimage as nd
from shapely.geometry import mapping
from osgeo import gdal
from sklearn.utils import shuffle
from scipy import misc


from DataHandle import NLCD_CLASSES, get_nlcd_stats
cid, nlcd_if, nlcd_dist, nlcd_var = get_nlcd_stats()


def color_aug(color):
    n_ch = color.shape[0]
    contra_adj = 0.05
    bright_adj = 0.05

    ch_mean = np.mean(color, axis=(-1,-2), keepdims=True).astype(np.float32)

    contra_mul = np.random.uniform(1-contra_adj, 1+contra_adj, (n_ch,1,1)).astype(np.float32)
    bright_mul = np.random.uniform(1-bright_adj, 1+bright_adj, (n_ch,1,1)).astype(np.float32)

    color = (color - ch_mean) * contra_mul + ch_mean * bright_mul
    return color


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Custom CNTK datasources
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

class MyDataSource(cntk.io.UserMinibatchSource):
    ''' A minibatch source for loading pre-extracted batches '''
    def __init__(self, f_dim, l_dim, m_dim, c_dim, highres_only, edge_sigma, edge_loss_boost,
            patch_list, num_highres=None, state_to_extract_highres_label=None):
        # Record passed parameters for use in later methods
        self.f_dim, self.l_dim, self.m_dim, self.c_dim = f_dim, l_dim, m_dim, c_dim
        self.highres_only = highres_only
        # ...and a few more that we can infer from the values that were passed
        self.num_color_channels, self.block_size, _ = self.f_dim
        self.lwm_dim = (1, self.l_dim[1], self.l_dim[2])
        self.edge_sigma = edge_sigma
        self.edge_loss_boost = edge_loss_boost
        self.num_nlcd_classes, self.num_landcover_classes = self.c_dim
        self.state_to_extract_highres_label = state_to_extract_highres_label

        # Record the stream information.
        self.fsi = cntk.io.StreamInformation(
            'features', 0, 'dense', np.float32, self.f_dim)
        self.lsi = cntk.io.StreamInformation(
            'landcover', 0, 'dense', np.float32, self.l_dim)
        self.lwi = cntk.io.StreamInformation(
            'lc_weight_map', 0, 'dense', np.float32, self.lwm_dim)
        self.msi = cntk.io.StreamInformation(
            'masks', 1, 'dense', np.float32, self.m_dim)
        self.csi = cntk.io.StreamInformation(
            'interval_centers', 1, 'dense', np.float32, self.c_dim)
        self.rsi = cntk.io.StreamInformation(
            'interval_radii', 1, 'dense', np.float32, self.c_dim)

        self.all_patches = [line.rstrip('\n') for line in open(patch_list)]
        t_highres_patches = [x for x in self.all_patches if \
                    int(os.path.basename(x).split('.npy')[0].split('-')[-2].split('_')[0]) <  100]
        t_superres_patches = [x for x in self.all_patches if \
                    int(os.path.basename(x).split('.npy')[0].split('-')[-2].split('_')[0]) == 100]

        self.num_highres = num_highres
        if self.num_highres is None: 
            if self.highres_only:
                self.all_patches = [x for x in self.all_patches if int(os.path.basename(x).split('.npy')[0].split('-')[-2].split('_')[0])<100]
            self.patch_list_for_all_classes = self.split_patch_per_class(self.all_patches)
        else:
            self.highres_patches = np.random.choice(t_highres_patches, self.num_highres, replace=False).tolist()
            self.superres_patches = t_superres_patches
            self.all_patches = self.highres_patches + self.superres_patches
            self.highres_patch_list_for_all_classes = self.split_patch_per_class(self.highres_patches)
            self.superres_patch_list_for_all_classes = self.split_patch_per_class(self.superres_patches)


        assert(len(self.all_patches) > 0)
        self.freq_control_arr = np.zeros((self.num_nlcd_classes,), dtype=np.float32)
        self.class_count_arr = np.zeros((self.num_nlcd_classes,), dtype=np.float32)
        self.batch_count = 0
        self.class_iter_idx = 0

        super(MyDataSource, self).__init__()

    def split_patch_per_class(self, all_patches):
        patch_list_for_all_classes = []
        for nlcd_class in range(self.num_nlcd_classes):
            patch_list_per_class = [x for x in all_patches if \
                        int(os.path.basename(x).split('.npy')[0].split('-')[-1].split('_')[nlcd_class])>0]
            #print('List len {} for class {}'.format(len(patch_list_per_class), nlcd_class))
            patch_list_for_all_classes.append(patch_list_per_class)
        return patch_list_for_all_classes

    def stream_infos(self):
        ''' Define the streams that will be returned by the minibatch source '''
        return [self.fsi, self.lsi, self.lwi, self.msi, self.csi, self.rsi]

    def to_one_hot(self, im, class_num):
        one_hot = np.zeros((class_num, im.shape[-2], im.shape[-1]), dtype=np.float32)
        for class_id in range(class_num):
            one_hot[class_id, :, :] = (im == class_id).astype(np.float32)
        return one_hot

    def sample_slices_from_list(self, patch_list):
        patch_file = random.sample(patch_list, 1)[0]
        state_str = patch_file.split('/')[-1].split('_')[0]

        minipatch = np.load(patch_file)
        while np.isnan(minipatch).any() or np.isinf(minipatch).any():
            logging.warning("Loaded one patch with nan or inf {}".format(patch_file))
            patch_file = random.sample(patch_list, 1)[0]
            minipatch = np.load(patch_file)
            state_str = patch_file.split('/')[-1].split('_')[0]

        naip_slice = minipatch[0, 0:4, ...]
        lc_slice = np.squeeze(minipatch[0, -2, ...])
        lc_slice[lc_slice>=4] = 4
        ##################################
        if self.state_to_extract_highres_label is not None:
            if state_str != self.state_to_extract_highres_label:
                lc_slice[:] = 0
        ##################################

        nlcd_slice = np.squeeze(minipatch[0, -1, ...])

        nlcd_class_count = np.zeros((self.num_nlcd_classes,), dtype=np.float32)
        for class_id in range(self.num_nlcd_classes):
            nlcd_class_count[class_id] = np.sum(nlcd_slice==class_id)
        return naip_slice, nlcd_slice, lc_slice, nlcd_class_count

    def print_class_count(self):
        self.batch_count += 1
        if self.batch_count % 200 == 0:
            class_percent = (0.5+self.class_count_arr/np.sum(self.class_count_arr) \
                    * 100).astype(np.uint8)
            logging.info("NLCD class counts: {}".format(class_percent))

    def get_random_instance(self):
        naip_slice, nlcd_slice, lc_slice, nlcd_class_count = \
            self.sample_slices_from_list(self.all_patches)
        return naip_slice, nlcd_slice, lc_slice

    def next_minibatch(self, mb_size, patch_freq_per_class):
        features = np.zeros((mb_size, self.num_color_channels,
            self.block_size, self.block_size), dtype=np.float32)
        landcover = np.zeros((mb_size, self.num_landcover_classes,
            self.block_size, self.block_size), dtype=np.float32)
        lc_weight_map = np.zeros((mb_size, 1,
            self.block_size, self.block_size), dtype=np.float32)
        masks = np.zeros((mb_size, self.num_nlcd_classes,
            self.block_size, self.block_size), dtype=np.float32)
        interval_centers = np.zeros(
            (mb_size, self.num_nlcd_classes, self.num_landcover_classes),
            dtype=np.float32)
        interval_radii = np.zeros(
            (mb_size, self.num_nlcd_classes, self.num_landcover_classes),
            dtype=np.float32)

        # Sample patches according to labels
        ins_id = 0
        while ins_id < mb_size:
            self.class_iter_idx = (self.class_iter_idx + 1) % len(patch_freq_per_class)
            patch_freq = patch_freq_per_class[self.class_iter_idx]

            self.freq_control_arr[self.class_iter_idx] += patch_freq
            while self.freq_control_arr[self.class_iter_idx] > 1 and ins_id < mb_size:
                self.freq_control_arr[self.class_iter_idx] -= 1

                if self.num_highres is None:
                    if len(self.patch_list_for_all_classes[self.class_iter_idx]) == 0:
                        continue
                    naip_slice, nlcd_slice, lc_slice, nlcd_class_count = \
                        self.sample_slices_from_list(self.patch_list_for_all_classes[self.class_iter_idx])
                else:
                    CALEB_PROB = 0.1
                    if np.random.rand() < CALEB_PROB: # sample from the highres patches with probability CALEB_PROB, else sample from superres
                        if len(self.highres_patch_list_for_all_classes[self.class_iter_idx]) == 0:
                            continue
                        naip_slice, nlcd_slice, lc_slice, nlcd_class_count = \
                            self.sample_slices_from_list(self.highres_patch_list_for_all_classes[self.class_iter_idx])                    
                    else:
                        if len(self.superres_patch_list_for_all_classes[self.class_iter_idx]) == 0:
                            continue
                        naip_slice, nlcd_slice, lc_slice, nlcd_class_count = \
                            self.sample_slices_from_list(self.superres_patch_list_for_all_classes[self.class_iter_idx])
                
                naip_slice = color_aug(naip_slice)
                self.class_count_arr += nlcd_class_count

                features[ins_id, :, :, :] = naip_slice
                landcover[ins_id, :, :, :] = self.to_one_hot(lc_slice, self.num_landcover_classes)
                lc_weight_map[ins_id, :, :, :] = 1.0
                masks[ins_id, :, :, :] = self.to_one_hot(nlcd_slice, self.num_nlcd_classes)
                interval_centers[ins_id, :, :] = nlcd_dist
                interval_radii[ins_id, :, :] = nlcd_var
                ins_id += 1
                
                
                # TODO: Usually 50% patches have high-res labels. The random sampling method below
                # will sample out a patch with high-res labels 50% of the time.
                # But since you (probably) have finished the other to-do item in this source code,
                # there might be just 1 to 256 patches with high-res labels.
                # The probablity of those patches being sampled out is very small.
                # So, change code here to make sure that with X% probability, a patch with highres labels
                # will be sampled out. Otherwise the little amount of high-res data will be effectively ignored.



                #naip_slice, nlcd_slice, lc_slice, nlcd_class_count = \
                #        self.sample_slices_from_list(self.patch_list_for_all_classes[self.class_iter_idx])

        self.print_class_count()
        result = {
            self.fsi: cntk.io.MinibatchData(cntk.Value(batch=features),
                        mb_size, mb_size, False),
            self.lsi: cntk.io.MinibatchData(cntk.Value(batch=landcover),
                        mb_size, mb_size, False),
            self.lwi: cntk.io.MinibatchData(cntk.Value(batch=lc_weight_map),
                        mb_size, mb_size, False),
            self.msi: cntk.io.MinibatchData(cntk.Value(batch=masks),
                        mb_size, mb_size, False),
            self.csi: cntk.io.MinibatchData(cntk.Value(batch=interval_centers),
                        mb_size, mb_size, False),
            self.rsi: cntk.io.MinibatchData(cntk.Value(batch=interval_radii),
                        mb_size, mb_size, False)
        }

        return result
