import torch
import numpy as np
import random, os, shutil
import matplotlib.pyplot as plt

"""
@uthor: Anthony Ortiz
Date: 03/25/2019
Last Modified: 03/25/2019
"""
class NamespaceFromDict(object):
    def __init__(self, d):
        self.__dict__.update(d)


def improve_reproducibility(seed=0):
    """Takes steps for reproducibility

    Args:
        seed(int): seed for the RNGs
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_all_rngs(seed)

def set_all_rngs(torch_rng_state, numpy_rng_state, python_rng_State):
    """Sets pytorch, numpy and python random generator states

    Args:
        torch_rng_state:  The torch rng state as returned by
            torch.get_rng_state()
        numpy_rng_state:  The numpy rng state as returned by
            np.random.get_state()
        python_rng_State: the python rng state as returned by random.getstate()
    """
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(numpy_rng_state)
    random.setstate(python_rng_State)

def seed_all_rngs(seed=0):
    """Seeds all supported random number generators

    Args:
        seed (int): seed for the rngs
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def saveLoss(train_loss, val_loss, save_dir, name = 'loss_plots'):
    """

    :param train_loss: train losses in different epochs
    :param val_loss: validation losses in different epochs
    :return:
    """
    plt.yscale('log')
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper right')
    plt.savefig(save_dir + name + '.png')

class CheckpointSaver(object):
    def __init__(self, save_dir, backup_dir):
        self.save_dir = save_dir
        self.backup_dir = backup_dir

    def save(self, state, is_best, checkpoint_name='checkpoint'):
        checkpoint_path = os.path.join(self.save_dir,
                                       '{}.pth.tar'.format(checkpoint_name))
        try:
            shutil.copyfile(
                checkpoint_path,
                '{}_bak'.format(checkpoint_path)
            )
        except IOError:
            pass
        torch.save(state, checkpoint_path)
        if is_best:
            try:
                shutil.copyfile(
                    os.path.join(self.backup_dir,
                                '{}_best.pth.tar'.format(checkpoint_name)),
                    os.path.join(self.backup_dir,
                                '{}_best.pth.tar_bak'.format(checkpoint_name))
                )
            except IOError:
                pass
            shutil.copyfile(
                checkpoint_path,
                os.path.join(self.backup_dir,
                             '{}_best.pth.tar'.format(checkpoint_name))
            )

