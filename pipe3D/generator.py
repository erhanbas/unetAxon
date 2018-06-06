import numpy as np
from pipe3D.utils import io_utils
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img

import os
import copy
import itertools

from .augment import augment_data, random_permutation_x_y

"""
Based on https://github.com/ellisdg/3DUnetCNN, 3d unet implementation trained on Bratts data. 
This one uses tensor flow by extending ch dimension in #/ch/z/y/x order
"""
def get_image(data, affine, nib_class=nib.Nifti1Image):
    return nib_class(dataobj=data, affine=affine)

def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)

def split_data(overwrite,split_file,num_sample,train_split_ratio,random_seed=None):
    # data split train/validation
    if not overwrite and os.path.exists(split_file):
        train_test_list = io_utils.pickle_load(split_file)
        train_list = train_test_list['train']
        test_list = train_test_list['test']
    else:
        sample_list = np.arange(num_sample)
        np.random.seed(random_seed)
        np.random.shuffle(sample_list)
        train_list = sample_list[0:round(num_sample * train_split_ratio)]
        test_list = sample_list[round(num_sample * train_split_ratio)::]
        train_test_list = {}
        train_test_list['train'] = train_list
        train_test_list['test'] = test_list
        io_utils.pickle_dump(train_test_list, split_file)
    return train_list, test_list

def fetch_data(input_raw_handle,input_label_handle,start,end,image_shape,augment=False):
    batch_x = input_raw_handle[start:end]
    batch_y = input_label_handle[start:end]
    raw_shape = batch_x.shape
    image_shape = np.asarray(image_shape)
    center_vox = np.round((np.asarray(raw_shape[2:])+1)/2)
    if np.any(image_shape != raw_shape[2:]):
        # crop image
        start_vox = np.asarray(center_vox - np.asarray(image_shape)/2,np.int)
        end_vox = np.asarray(start_vox + image_shape,np.int)
        batch_x = batch_x[:,:,
                   start_vox[0]:end_vox[0],
                   start_vox[1]:end_vox[1],
                   start_vox[2]:end_vox[2]]
        batch_y = batch_y[:,
                   start_vox[0]:end_vox[0],
                   start_vox[1]:end_vox[1],
                   start_vox[2]:end_vox[2]]

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

def custom_generator(input_raw_handle,input_label_handle, index_list, image_shape, batch_size, labels=None, augment=False):
    """ creates a generator to iterate on training data in batches"""
    len_list = len(index_list)
    batch_size = np.minimum(len_list,batch_size)
    start_end = list(range(0, len_list - 1, batch_size)) + [len_list]
    affine = np.eye(4, 4)
    for i_list in range(len(start_end)-1):
        # print(i_list,start_end[i_list],start_end[i_list+1])
        start = start_end[i_list]
        end = start_end[i_list+1]
        batch_x, batch_y = fetch_data(input_raw_handle,input_label_handle, start,end,image_shape,augment)
        batch_y = batch_y[:,np.newaxis] # extend in channel direction
        if augment:
            """Augment data with flip/rotation/scale/etc..."""
            scale, flip, rotation = None
            batch_x, batch_y = augment_data(batch_x,batch_y,scale,flip,rotation)

        # convert to categorical variables, extend 1st dim for tf, last dim for th
        batch_y = to_categorical(batch_y)
        yield batch_x, batch_y

def get_generators(input_raw_handle, input_label_handle, batch_size, image_shape, split_file,
                   train_split_ratio=0.66, overwrite=False, labels=None, augment=False,
                   augment_flip=True, augment_distortion_factor=0.25, permute=False):

    num_sample, num_channel, depth, height, width = input_raw_handle.shape
    train_list, test_list = split_data(overwrite, split_file, num_sample, train_split_ratio=train_split_ratio, random_seed=35)

    # training generator
    train_generator = custom_generator(input_raw_handle,input_label_handle, train_list, image_shape, batch_size,augment=augment)
    # testing generator
    validation_generator = custom_generator(input_raw_handle,input_label_handle, test_list, image_shape, batch_size,augment=augment)

    n_train_steps = get_num_step(len(train_list),batch_size)
    n_validation_steps = get_num_step(len(test_list),batch_size)

    return train_generator, validation_generator, n_train_steps, n_validation_steps

def get_num_step(siz,batch_size):
    if siz<batch_size:
        num_step = 1
    elif not siz%batch_size:
        num_step = siz // batch_size
    else:
        num_step = siz // batch_size + 1

    return num_step

