from __future__ import print_function
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
#Import our classes
from nets.unet import generate_batch_norm_unet
from nets.unet3d import generate_unet, generate_3D_unet
from utils.image import ImageDataGenerator
from utils.dice import dice_coef
import h5py
import os


#Import specific keras classes
from keras.optimizers import Adam
from keras.callbacks import  ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.utils import to_categorical


#Make sure we remove any randomness
from numpy.random import seed
seed(1)

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

input_folder = '/Users/base/Dropbox (HHMI)/DATA/annotated_neuron'

#UNCOMMENT BELOW TO DETERMINE GPU
# from keras import backend as K
# import tensorflow as tf
# import os
# #Use one GPU
# if K.backend() == 'tensorflow':
#     # Use only gpu #X (with tf.device(/gpu:X) does not work)
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     # Automatically choose an existing and supported device if the specified one does not exist
#     config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#     # To constrain the use of gpu memory, otherwise all memory is used
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#     K.set_session(sess)
###############################################################################################

# #Show an originalimage
# image = cv2.imread('data/training/training_images/images/MCUCXR_0001_0.png',0)
# plt.figure(figsize=(12,6))
# plt.imshow(image, cmap=cm.Greys_r, interpolation='none')
# plt.axis('off')
#
# #Show corresponding mask
# mask = cv2.imread('data/training/training_masks/images/MCUCXR_0001_0.png',0)
# plt.figure(figsize=(12,6))
# plt.imshow(mask, cmap=cm.Greys_r, interpolation='none')
# plt.axis('off')
# print ('Image shape: ',image.shape)
# print ('Mask shape: ',mask.shape)

# inputfile_path = 'r"'+os.path.join(input_folder,'2017-09-25_G-007_consensus-annotation.h5')+'"'

input_f = '/Users/base/Dropbox (HHMI)/DATA/annotated_neuron/2017-09-25_G-007_consensus-training_dense_label.h5'
input_f = input_f.replace('/','//')
inputfile_handle = h5py.File(input_f,'r')
d_set = inputfile_handle['volume']

binary_mask = to_categorical(d_set, num_classes=2)

print ('Binary shape: ',binary_mask.shape)

plt.figure(figsize=(12,6))
ax = plt.subplot(1, 3, 1)
ax.set_title('Background')
ax.imshow(binary_mask[...,0], cmap=cm.Greys_r, interpolation='none')
ax.axis('off')
ax = plt.subplot(1, 3, 2)
ax.set_title('Right Lung')
ax.imshow(binary_mask[...,1], cmap=cm.Greys_r, interpolation='none')
ax.axis('off')
ax = plt.subplot(1, 3, 3)
ax.set_title('Left Lung')
ax.imshow(binary_mask[...,2], cmap=cm.Greys_r, interpolation='none')
ax.axis('off')


def custom_image_generator(generator, directory, seed=None, batch_size=16, target_size=(128, 128),
                           color_mode="grayscale", class_mode=None, isMask=False, num_classes=None):
    """
    Read images from a dirctory batch-size wise
    If images are masks (e.g. 128x128x1) then convert them to multi-label arrays (e.g. 128x128x3) so that they match the output of UNet
    """
    import numpy as np

    # Read from directory (flow_from_directory)
    iterator = generator.flow_from_directory(directory=directory,
                                             target_size=target_size,
                                             color_mode=color_mode,
                                             class_mode=class_mode,
                                             batch_size=batch_size,
                                             seed=seed,
                                             shuffle=True)

    for batch_x in iterator:
        # if image is a mask convert to a multi-label array (binary matrix: 128x128x3)
        if isMask == True:
            batch_x = to_categorical(batch_x, num_classes)
        yield batch_x


#Random seed set into a fixed value to apply same augmentations to images and masks. Otherwise they would not be the same.
seed=1

#Set the batch size
batch_size=4


#Create the image training generator
image_train_datagen = custom_image_generator(
            ImageDataGenerator(rotation_range=10.,  #Augmentation 1: Rotate images randomly within +- 20 degrees
                     width_shift_range=0.1, #Augmentation 2: Translate image left or right by 10%
                     height_shift_range=0.1,  #Augmentation 3: Translate image up or down by 10%
                     zoom_range=0.2, rescale=1./255.), #Augmentation 4: Zoom in/out 20%
            directory='data/training/training_images/',  #Directory holding the raw images
            seed=seed, #Use a specific random seed
            target_size=(128,128), #Resize images if needed to fit into the Input layer of Unet
            color_mode='grayscale', #Load them as one-channel (i.e. grayscale)
            batch_size=batch_size, #Use batch size of 32
        )

#Create the mask training generator
mask_train_datagen = custom_image_generator(
            ImageDataGenerator(rotation_range=10.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2),
            directory='data/training/training_masks/',
            seed=seed,
            target_size=(128,128),
            color_mode='grayscale',
            batch_size=batch_size,
            isMask=True
)


#Create the image validation generator
image_val_datagen = custom_image_generator(
            ImageDataGenerator(rescale=1./255.),
            directory='data/validation/validation_images/',
            seed=seed,
            target_size=(128,128),
            color_mode='grayscale',
            batch_size=batch_size,
        )

#Create the mask validation generator
mask_val_datagen = custom_image_generator(
            ImageDataGenerator(),
            directory='data/validation/validation_masks/',
            seed=seed,
            target_size=(128,128),
            color_mode='grayscale',
            batch_size=batch_size,
            isMask=True
        )



# combine generators into one which yields image and images
train_generator = zip(image_train_datagen, mask_train_datagen)
val_generator = zip(image_val_datagen, mask_val_datagen)


#Generate UNet with base number of filters and number of labels for segmntation
# model = generate_batch_norm_unet(base_num_filters=16, num_classes=3, kernel_size=(10,10))
model = generate_3D_unet(base_num_filters=16, num_classes=3, kernel_size=(5,5,5))

print (model.summary())