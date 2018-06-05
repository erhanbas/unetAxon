from __future__ import print_function
import glob
import socket
import os
import h5py

from pipe3D.myconfig import initconfig
config = initconfig()

from pipe3D.model import unet3D
from pipe3D.generator import get_generators


from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
# from unet3d.model import unet_model_3d
from unet3d.training import load_old_model, train_model


#
#
# # %matplotlib inline
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import tensorflow
#
# # import cv2
# import numpy as np
# #Import our classes
# from nets.unet import generate_batch_norm_unet
# from nets.unet3d import generate_unet, generate_3D_unet
# from utils.image import ImageDataGenerator
# from utils import custom_generator
# from utils.multi_gpu import make_parallel
# from utils.dice import dice_coef

# from keras.preprocessing.image import ImageDataGenerator as IDG
# #Import specific keras classes
# from keras.optimizers import Adam
# from keras.callbacks import  ModelCheckpoint, CSVLogger, ReduceLROnPlateau
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
#
#
# #Make sure we remove any randomness
# from numpy.random import seed
# seed(1)
#
# try:
#     from itertools import izip as zip
# except ImportError: # will be 3.x series
#     pass

#UNCOMMENT BELOW TO DETERMINE GPU
# from keras import backend as K
# import tensorflow as tf
# import os
#Use one GPU
# if K.backend() == 'tensorflow':
#     # Use only gpu #X (with tf.device(/gpu:X) does not work)
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
#     # Automatically choose an existing and supported device if the specified one does not exist
#     config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#     # To constrain the use of gpu memory, otherwise all memory is used
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#     K.set_session(sess)
###############################################################################################

def setup_paths():
    # setup paths/environments
    # set the path based on machine
    if socket.gethostname() == 'base-ws1':
        datafold = '/data2/Dropbox (HHMI)/DATA/annotated_neuron'
    elif socket.gethostname() == 'vega':
        # do nothing
        1
    else:
        # do nothing
        datafold = '/Users/base/Dropbox (HHMI)/DATA/annotated_neuron'
    return datafold
def fetch_train_data(input_h5_file='2017-09-25_G-007_consensus-training_raw.h5:volume'):
    # load data
    datafold = setup_paths()
    # check for dataset
    file,dataset = input_h5_file.split(':')
    input_raw_f = os.path.join(datafold, file).replace('/', '//')
    input_raw_handle = h5py.File(input_raw_f, 'r')[dataset]
    return input_raw_handle

def main():
    overwrite = True
    # training data handle
    input_raw_handle = fetch_train_data(config['data_file'])
    input_mask_handle = fetch_train_data(config['label_file'])

    # load model
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = unet3D.generate_unet3D(input_shape=config["input_shape"],
                                   pool_size=config["pool_size"],
                                   n_labels=config["n_labels"],
                                   initial_learning_rate=config["initial_learning_rate"],
                                   deconvolution=config["deconvolution"])
    print(model.summary())

    # generators for training and validation with augmentation
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_generators(
        input_raw_handle,
        input_mask_handle,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])


    # image_gen = IDG(featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True)
    #
    # image_train_datagen = custom_generator.custom_image_generator(image_gen, X_train, y_train, seed, batch_size=16)
    # image_validation_datagen = custom_generator.custom_image_generator(image_gen, X_test, y_test, seed, batch_size=16)
    #
    #
    # # model = generate_3D_unet(base_num_filters=16, num_classes=3, kernel_size=(5, 5, 5))
    #
    #
    # # d_set_raw = input_raw_handle['volume'][:,0,:,:,:]
    # # d_set_raw = np.reshape(d_set_raw,d_set_raw.shape+(1,))
    # # input_mask_f = os.path.join(datafold,'annotated_neuron/2017-09-25_G-007_consensus-training_dense_label.h5')
    # # input_mask_f = input_mask_f.replace('/','//')
    # # input_mask_handle = h5py.File(input_mask_f,'r')
    # binary_mask = to_categorical(input_mask_handle['volume'], num_classes=3)
    # print ('Binary shape: ',binary_mask.shape)
    #
    #
    #
    #
    # X_train, X_test, y_train, y_test = train_test_split(d_set_raw, binary_mask, test_size=0.33, random_state=42)
    # ## ####################################################
    #
    # # plt.figure(figsize=(12,6))
    # # ax = plt.subplot(1, 3, 1)
    # # ax.set_title('Background')
    # # ax.imshow(np.max(binary_mask[0,:,:,:,:],axis=2), cmap=cm.Greys_r, interpolation='none')
    # # ax.axis('off')
    #
    #
    # #Random seed set into a fixed value to apply same augmentations to images and masks. Otherwise they would not be the same.
    # seed=1
    # #Set the batch size
    # batch_size=4
    #
    # # numsamples = d_set_raw.shape[0]
    # # randsamples = np.random.choice(numsamples,numsamples)
    # # data_x_train = d_set_raw[0:numsamples//10*9,...]
    # # data_y_train = binary_mask[::numsamples//10*9,...]
    # #
    # # data_x_validation = d_set_raw[::numsamples//10*9+1]
    # # data_y_validation = binary_mask
    #
    # ######################################################################
    #
    # image_gen = IDG(featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True)
    #
    # image_train_datagen = custom_generator.custom_image_generator(image_gen, X_train, y_train, seed, batch_size=16)
    # image_validation_datagen = custom_generator.custom_image_generator(image_gen, X_test, y_test, seed, batch_size=16)
    #
    # # combine generators into one which yields image and images
    # # train_generator = zip(image_train_datagen, mask_train_datagen)
    # # val_generator = zip(image_val_datagen, mask_val_datagen)
    #
    #
    # #Generate UNet with base number of filters and number of labels for segmntation
    # # model = generate_batch_norm_unet(base_num_filters=16, num_classes=3, kernel_size=(10,10))
    # model = generate_3D_unet(base_num_filters=16, num_classes=3, kernel_size=(5,5,5))
    # # model2 = ge
    #
    # print (model.summary())
    #
    # #Save UNet model on smallest validation loss
    # model_checkpoint = ModelCheckpoint('./models/3d_unet_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # reducer = ReduceLROnPlateau(monitor='loss', factor=.8, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    # #Save logs to file
    # logger = CSVLogger('experiment.txt', separator=',', append=False)
    # #Put the 2 callbacks together
    # callbacks = [model_checkpoint,reducer, logger]
    #
    # # # Goal: make the more frequent labels weigh less
    # # class_frequencies = np.array([1221018,  221993,  195389])
    # # class_weights = class_frequencies.sum() / class_frequencies.astype(np.float32)
    # # class_weights = class_weights ** 0.5
    # # print (class_weights)
    #
    # train_steps_per_epoch = round(X_train.shape[0]/batch_size)
    # val_steps_per_epoch = round(X_test.shape[0]/batch_size)
    #
    # def dice_error(y_true, y_pred):
    #     return 1-dice_coef(y_true, y_pred)
    #
    # model.compile(optimizer=Adam(lr=1e-3), loss=[dice_error], metrics=[dice_coef], sample_weight_mode='temporal')
    # model.fit_generator(image_train_datagen, steps_per_epoch=train_steps_per_epoch, epochs=150,
    #                     validation_data=image_validation_datagen, validation_steps=val_steps_per_epoch,
    #                     callbacks=callbacks, verbose=1)
    # # model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=150,
    # #                     validation_data=val_generator, validation_steps=val_steps_per_epoch,
    # #                     class_weight=class_weights, callbacks=callbacks, verbose=0)


if __name__ == "__main__":
    main()