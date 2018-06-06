import numpy as np
from pipe3D.utils import io_utils

import os
import copy
import itertools

from .augment import augment_data, random_permutation_x_y

"""
Based on https://github.com/ellisdg/3DUnetCNN, 3d unet implementation trained on Bratts data. 
This one uses tensor flow by extending ch dimension in #/ch/z/y/x order
"""
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

def fetch_data(input_raw_handle,input_label_handle,start,end,augment=False):
    batch_x = input_raw_handle[start:end]
    batch_y = input_label_handle[start:end]
    return batch_x, batch_y

def to_categorical(ylabels,labels=None):
    inshape = list(ylabels.shape)
    if not labels:
        labels = np.setdiff1d(np.unique(ylabels),0)
    n_labels = len(labels)
    inshape[1] = n_labels
    ylabels_out = np.zeros(inshape)
    if n_labels == 1:
        ylabels_out[ylabels>0] = 1
    elif n_labels > 1:
        for i_lab in range(n_labels):
            ylabels_out[:,i_lab][ylabels==labels[i_lab]] = 1

    return ylabels_out

def custom_generator(input_raw_handle,input_label_handle, index_list, batch_size, labels=None, augment=False):
    """ creates a generator to iterate on training data in batches"""

    len_list = len(index_list)
    batch_size = np.minimum(len_list,batch_size)
    start_end = list(range(0, len_list - 1, batch_size)) + [len_list]
    for i_list in range(len(start_end)-1):
        # print(i_list,start_end[i_list],start_end[i_list+1])
        start = start_end[i_list]
        end = start_end[i_list+1]
        batch_x, batch_y = fetch_data(input_raw_handle,input_label_handle, start,end)
        batch_y = batch_y[:,np.newaxis] # extend in channel direction
        if augment:
            """Augment data with flip/rotation/scale/etc..."""
            scale, flip, rotation = None
            batch_x, batch_y = augment_data(batch_x,batch_y,scale,flip,rotation)

        # convert to categorical variables, extend 1st dim for tf, last dim for th
        batch_y = to_categorical(batch_y)
        yield batch_x, batch_y

def get_generators(input_raw_handle, input_label_handle, batch_size, n_labels, split_file,
                   test_size=0.33, overwrite=False, labels=None, augment=False,
                   augment_flip=True, augment_distortion_factor=0.25, patch_shape=None,
                   validation_patch_overlap=0, training_patch_start_offset=None,
                   validation_batch_size=None, skip_blank=True, permute=False):
    train_generator, validation_generator, n_train_steps, n_validation_steps = None

    num_sample, num_channel, depth, height, width = input_raw_handle.shape
    train_list, test_list = split_data(overwrite, split_file, num_sample, test_size=0.33, random_seed=35)

    # training generator
    train_generator = custom_generator(input_raw_handle,input_label_handle, train_list, batch_size,augment=False)
    # testing generator
    validation_generator = custom_generator(input_raw_handle,input_label_handle, test_list, batch_size,augment=False)

    n_train_steps = get_num_step(train_list,batch_size)
    n_validation_steps = get_num_step(test_list,batch_size)

    return train_generator, validation_generator, n_train_steps, n_validation_steps

def get_num_step(siz,batch_size):
    if siz<batch_size:
        num_step = 1
    elif not siz%batch_size:
        num_step = siz // batch_size
    else:
        num_step = siz // batch_size + 1

    return num_step

