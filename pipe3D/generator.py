import numpy as np
from utils import io_utils


import os
import copy
import itertools

from .augment import augment_data, random_permutation_x_y

"""Based on https://github.com/ellisdg/3DUnetCNN, 3d unet implementation trained on Bratts data"""
def split_data(overwrite,split_file,num_sample,test_size=0.33,random_seed=None):
    # data split train/validation
    if overwrite or not os.path.exists(split_file):
        train_test_list = io_utils.pickle_load(split_file)
        train_list = train_test_list['train']
        test_list = train_test_list['test']
    else:
        sample_list = np.arange(num_sample)
        np.random.seed(random_seed)
        np.random.shuffle(sample_list)
        train_list = sample_list[0:round(num_sample * test_size)]
        test_list = sample_list[round(num_sample * test_size)::]
        train_test_list = {}
        train_test_list['train'] = train_list
        train_test_list['test'] = test_list
        io_utils.pickle_dump(train_test_list, split_file)
    return train_list, test_list


def get_generators(input_raw_handle, input_mask_handle, batch_size, n_labels, split_file,
                   test_size=0.33, overwrite=False, labels=None, augment=False,
                   augment_flip=True, augment_distortion_factor=0.25, patch_shape=None,
                   validation_patch_overlap=0, training_patch_start_offset=None,
                   validation_batch_size=None, skip_blank=True, permute=False):
    train_generator, validation_generator, n_train_steps, n_validation_steps = None
    num_sample, num_channel, depth, height, width = input_raw_handle.shape
    train_list, test_list = split_data(overwrite, split_file, num_sample, test_size=0.33, random_seed=35)

    # training generator
    train_generator = data_generator()

    # testing generator



    return train_generator, validation_generator, n_train_steps, n_validation_steps
