import argparse
import json, os
from pytorch.models.unet import Unet
from pytorch.models.fusionnet import Fusionnet
from pytorch.models.conditional_superres_net import Conditional_superres_net
from pytorch.losses import (multiclass_ce, multiclass_dice_loss, multiclass_jaccard_loss, multiclass_tversky_loss)
from pytorch.train import train, Framework, train_superres
from pytorch.data_loader import DataGenerator
from torch.utils import data


"""
@uthor: Anthony Ortiz
Date: 03/25/2019
Last Modified: 03/25/2019
"""
def parse_int_list(s):
    if s == '': return ()
    return tuple(int(n) for n in s.split(','))



parser = argparse.ArgumentParser()

### JSON FILE IF NEW EXPERIMENT
parser.add_argument('--config_file', type=str,
                    help="json file containing the configuration")


### GENERAL EXPERIMENT SETTINGS
parser.add_argument('--debug', action='store_true', help=(
    "whether to log debug information, such as norms of weights and gradients."
))

args = parser.parse_args()

assert args.config_file is not None
params = json.load(open(args.config_file))

training_patches_fn = params["training_patches_fn"]
validation_patches_fn = params["validation_patches_fn"]
f = open(training_patches_fn, "r")
training_patches = f.read().strip().split("\n")
f.close()

f = open(validation_patches_fn, "r")
validation_patches = f.read().strip().split("\n")
f.close()

batch_size = params["loader_opts"]["batch_size"]
patch_size = params["patch_size"]
num_channels = params["loader_opts"]["num_channels"]
params_train = {'batch_size': params["loader_opts"]["batch_size"],
          'shuffle': params["loader_opts"]["shuffle"],
          'num_workers': params["loader_opts"]["num_workers"]}

training_set = DataGenerator(
        training_patches, batch_size, patch_size, num_channels, superres=params["train_opts"]["superres"]
    )

validation_set = DataGenerator(
        validation_patches, batch_size, patch_size, num_channels, superres=params["train_opts"]["superres"]
    )

def patch_gen_train():
    return data.DataLoader(training_set, **params_train)

def patch_gen_val():
    return data.DataLoader(validation_set, **params_train)
def main():
    train_opts = params["train_opts"]
    model_opts = params["model_opts"]

    # Default model is Duke_Unet

    if model_opts["model"] == "unet":
        model = Unet
    elif model_opts["model"] == "conditional_superres_net":
        model = Conditional_superres_net
    elif model_opts["model"] == "fusionnet":
        model = Fusionnet
    else:
        print(
            "Option {} not supported. Available options: unet, conditional_superres_net".format(
                model_opts["model"]))
        raise NotImplementedError

    if train_opts["loss"] == "dice":
        loss = multiclass_dice_loss
    elif train_opts["loss"] == "ce":
        loss = multiclass_ce
    elif train_opts["loss"] == "jaccard":
        loss = multiclass_jaccard_loss
    elif train_opts["loss"] == "tversky":
        loss = multiclass_tversky_loss
    else:
        print("Option {} not supported. Available options: dice, ce, jaccard, tversky".format(train_opts["loss"]))
        raise NotImplementedError

    frame = Framework(
        model(model_opts),
        loss(),
        train_opts["optimizer_lr"]
    )
    if not os.path.exists(params["save_dir"]):
        os.makedirs(params["save_dir"])

    dataloaders = {'train': patch_gen_train, 'val': patch_gen_val}

    #FIXME: Not sure if shuffling is working

    if model_opts["model"] == "conditional_superres_net":
        _, train_history, val_history = train_superres(frame, dataloaders, train_opts["n_epochs"],
                                              params)
    else:
        _, train_history, val_history = train(frame, dataloaders, train_opts["n_epochs"],
                                              params)



if __name__ == "__main__":
    main()

