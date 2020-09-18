# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Implementation of ModelSessionAbstract for the Orinoquia land cover mapping project. Heavily references
wildlife-conservation-society.orinoquia-land-use/training/inference.py

"wildlife-conservation-society.orinoquia-land-use" and "ai4eutils" need to be on the PYTHONPATH.
"""

import importlib
import logging
import sys
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from web_tool.ModelSessionAbstract import ModelSession

LOGGER = logging.getLogger("server")


class TorchFineTuningOrinoquia(ModelSession):

    def __init__(self, gpu_id, **kwargs):
        # setting up device to use
        LOGGER.debug(f"TorchFineTuningOrinoquia init, gpu_id is {gpu_id}")
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu")

        # load experiment configuration as a module
        config_module_path = kwargs["config_module_path"]
        try:
            module_name = "config"
            spec = importlib.util.spec_from_file_location(module_name, config_module_path)
            self.config = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = self.config
            spec.loader.exec_module(self.config)
        except Exception as e:
            LOGGER.error(f"Failed to import experiment and model configuration. Exception: {e}")
            sys.exit(1)
        LOGGER.info(f"config is for experiment {self.config.experiment_name}")

        # check that the necessary fields are present in the config
        assert self.config.num_classes > 1
        assert self.config.chip_size > 1
        assert self.config.feature_scale in [1, 2]

        chip_size = self.config.chip_size
        self.prediction_window_size = self.config.prediction_window_size if self.config.prediction_window_size else 128
        self.prediction_window_offset = int((chip_size - self.prediction_window_size) / 2)
        print((f"Using chip_size {chip_size} and window_size {self.prediction_window_size}. "
               f"So window_offset is {self.prediction_window_offset}"))

        # obtain the model that the config has initialized
        self.checkpoint_path = kwargs["fn"]
        assert os.path.exists(self.checkpoint_path), f"Checkpoint at {self.checkpoint_path} does not exist."
        self.model = self.config.model
        self._init_model()

        # other instance variables
        self._last_tile = None

        # recording the current feature map (before final layer), and the corrections made
        self.current_features = None
        self.corr_features = []
        self.corr_labels = []

    def _init_model(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        print(f"Using checkpoint at epoch {checkpoint['epoch']}, step {checkpoint['step']}, "
              f"val accuracy is {checkpoint.get('val_acc', 'Not Available')}")
        self.model = self.model.to(device=self.device)
        self.model.eval()

    @property
    def last_tile(self):
        return self._last_tile

    def run(self, tile, inference_mode=False):
        assert tile.shape[2] == 11  # 10 Landsat bands + DEM
        height = tile.shape[0]  # tile is of dims (height, width, channels)
        width = tile.shape[1]
        self._last_tile = tile
        prediction_window_offset = self.prediction_window_offset
        prediction_window_size = self.prediction_window_size

        # apply the preprocessing of bands to the tile
        data_array = self.config.preprocess_tile(tile)  # dim is (6, H, W)

        LOGGER.debug(f"run, tile shape is {tile.shape}")
        LOGGER.debug(f"run, data_array shape is {data_array.shape}")

        # pad by mirroring at the edges to predict only on the center crop
        data_array = np.pad(data_array,
                            [
                                (0, 0),  # only pad height and width
                                (prediction_window_offset, prediction_window_offset),  # height / rows
                                (prediction_window_offset, prediction_window_offset)  # width / cols
                            ],
                            mode="symmetric")

        # form batches
        batch_size = self.config.batch_size if self.config.batch_size is not None else 32
        batch = []
        batch_indices = []  # cache these to save recalculating when filling in model predictions

        chip_size = self.config.chip_size

        num_rows = math.ceil(height / prediction_window_size)
        num_cols = math.ceil(width / prediction_window_size)

        for col_idx in range(num_cols):
            col_start = col_idx * prediction_window_size
            col_end = col_start + chip_size

            for row_idx in range(num_rows):
                row_start = row_idx * prediction_window_size
                row_end = row_start + chip_size

                chip = data_array[:, row_start:row_end, col_start: col_end]
                # pad to (chip_size, chip_size)
                chip = np.pad(chip,
                              [(0, 0), (0, chip_size - chip.shape[1]), (0, chip_size - chip.shape[2])])

                # processing it as the dataset loader _get_chip does
                chip = np.nan_to_num(chip, nan=0.0, posinf=1.0, neginf=-1.0)
                sat_mask = chip[0].squeeze() > 0.0  # mask out DEM data where there's no satellite data
                chip = chip * sat_mask

                batch.append(chip)

                valid_row_end = row_start + min(prediction_window_size, height - row_idx * prediction_window_size)
                valid_col_end = col_start + min(prediction_window_size, width - col_idx * prediction_window_size)
                batch_indices.append(
                    (row_start, valid_row_end, col_start, valid_col_end))  # explicit to be less confusing
        batch = np.array(batch)  # (num_chips, channels, height, width)

        # score chips in batches
        model_output = []
        model_features = []  # same spatial dims as model_output, but has 64 or 32 channels
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(batch), batch_size):
                t_batch = batch[i:i + batch_size]
                t_batch = torch.from_numpy(t_batch).to(self.device)

                scores, features = self.model.forward(t_batch,
                                                      return_features=True)  # these are scores before the final softmax
                softmax_scores = torch.nn.functional.softmax(scores,
                                                             dim=1).cpu().numpy()  # (batch_size, num_classes, H, W)

                softmax_scores = np.transpose(softmax_scores, axes=(0, 2, 3, 1))  # (batch_size, H, W, num_classes)
                # only save the center crop
                softmax_scores = softmax_scores[
                                 :,
                                 prediction_window_offset:prediction_window_offset + prediction_window_size,
                                 prediction_window_offset:prediction_window_offset + prediction_window_size,
                                 :
                                 ]
                model_output.append(softmax_scores)

                features = features.cpu().numpy()
                features = np.transpose(features, axes=(0, 2, 3, 1))  # (batch_size, H, W, num_features)
                features = features[
                           :,
                           prediction_window_offset:prediction_window_offset + prediction_window_size,
                           prediction_window_offset:prediction_window_offset + prediction_window_size,
                           :
                           ]
                model_features.append(features)

        model_output = np.concatenate(model_output, axis=0)
        model_features = np.concatenate(model_features, axis=0)

        # fill in the output array
        output = np.zeros((height, width, self.config.num_classes), dtype=np.float32)
        for i, (row_start, row_end, col_start, col_end) in enumerate(batch_indices):
            h = row_end - row_start
            w = col_end - col_start
            output[row_start:row_end, col_start:col_end, :] = model_output[i, :h, :w, :]
        print(f"--- Orinoquia ModelSession, output[-1, -1, :] is {output[-1, -1, :]}")

        num_features = 64 if self.config.feature_scale == 1 else 32
        output_features = np.zeros((height, width, num_features), dtype=np.float32)  # float32 used during training too
        for i, (row_start, row_end, col_start, col_end) in enumerate(batch_indices):
            h = row_end - row_start
            w = col_end - col_start
            output_features[row_start:row_end, col_start:col_end, :] = model_features[i, :h, :w, :]
        print(f"--- Orinoquia ModelSession, output_features[-1, -1, :] is {output_features[-1, -1, :]}")
        # save the features
        self.current_features = output_features

        return output

    def add_sample_point(self, row, col, class_idx):
        self.corr_labels.append(class_idx)
        self.corr_features.append(self.current_features[row, col, :])
        print(f"After add_sample_point, corr_labels length is {len(self.corr_labels)}")
        return {
            "message": f"Training sample for class {class_idx} added",
            "success": True
        }

    def undo(self):
        if len(self.corr_features) > 0:
            self.corr_features = self.corr_features[:-1]
            self.corr_labels = self.corr_labels[:-1]
            return {
                "message": "Undid training sample",
                "success": True
            }
        else:
            return {
                "message": "Nothing to undo",
                "success": False
            }

    def reset(self):
        self._init_model()
        self.corr_features = []
        self.corr_labels = []
        return {
            "message": "Model is reset",
            "success": True
        }

    def retrain(self, train_steps=100, learning_rate=5e-5):
        print("In retrain...")
        print_every = 10

        # all corrections since the last reset are used
        batch_x = torch.from_numpy(np.array(self.corr_features)).float().to(self.device)
        batch_y = torch.from_numpy(np.array(self.corr_labels)).to(self.device)

        # make the last layer `final` trainable TODO do we need to do this - default trainable?
        for param in self.model.final.parameters():  # see UNet implementation in WCS project repo
            param.requires_grad = True

        optimizer = optim.Adam(self.model.final.parameters(), lr=learning_rate)  # only the last layer

        # during re-training, we use equal weight for all classes
        criterion = nn.CrossEntropyLoss().to(device=self.device)

        self.model.train()
        for step in range(train_steps):
            with torch.enable_grad():
                # forward pass
                batch_x_reshaped = batch_x.unsqueeze(2).unsqueeze(3)
                scores = self.model.final.forward(batch_x_reshaped).squeeze(3).squeeze(2)
                loss = criterion(scores, batch_y)

                # backward pass
                optimizer.zero_grad()
                loss.backward()  # compute gradients
                optimizer.step()  # update parameters

                if step % print_every == 0:
                    preds = scores.argmax(1)
                    accuracy = (batch_y == preds).float().mean()

                    print(f'step {step}, loss: {loss.item()}, accuracy: {accuracy.item()}')

        return {
            "message": f"Fine-tuned model with {len(self.corr_features)} samples for {train_steps} steps",
            "success": True
        }

    def save_state_to(self, directory):
        return {
            "message": "Saving not yet implemented",
            "success": False
        }

    def load_state_from(self, directory):
        return {
            "message": "Saving and loading not yet implemented",
            "success": False
        }
