from __future__ import print_function
import glob
import os, sys, getopt

from __myconfig import initconfig
config = initconfig()
from unet.model import unet3D
from unet.generator import get_generators
from unet.training import load_old_model, train_model
from utils.io_utils import preload_data
from keras.utils import plot_model
import numpy as np


# from sklearn.model_selection import train_test_split
# #Make sure we remove any randomness
# from numpy.random import seed
# seed(1)
#
# try:
#     from itertools import izip as zip
# except ImportError: # will be 3.x series
#     pass

# UNCOMMENT BELOW TO DETERMINE GPU
from keras import backend as K
import tensorflow as tf
## Use one GPU
# if K.backend() == 'tensorflow':
#     # Use only gpu #X (with tf.device(/gpu:X) does not work)
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
#     # Automatically choose an existing and supported device if the specified one does not exist
#     config_GPU = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#     # To constrain the use of gpu memory, otherwise all memory is used
#     config_GPU.gpu_options.allow_growth = True
#     sess = tf.Session(config=config_GPU)
#     K.set_session(sess)
###############################################################################################


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "hc:", ["cnum="])
    except getopt.GetoptError:
        gpu_id = None
        # print('train.py -c <num_gpu>')
        # sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('train.py -c <>')
            sys.exit()
        elif opt in ("-c", "--cnum"):
            gpu_id = arg

    overwrite = True
    # training data handle
    input_raw_handle = preload_data(config['data_file'])
    input_label_handle = preload_data(config['label_file'])

    if gpu_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    # load model
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = unet3D.generate_unet3D(input_shape=config["input_shape"],
                                   pool_size=config["pool_size"],
                                   n_labels=config["n_labels"],
                                   initial_learning_rate=config["initial_learning_rate"],
                                   deconvolution=config["deconvolution"])
    print(model.summary())

    # generators for training and validation with augmentation

    # class_frequencies = np.array([np.prod(input_label_handle.shape)-np.sum(input_label_handle[:]>0),np.sum(input_label_handle[:]>0)])
    # class_weights = class_frequencies.sum() / class_frequencies.astype(np.float32)
    # # class_weights = class_weights ** 0.5
    class_weights = None
    # print (class_weights)

    train_generator, validation_generator, n_train_steps, n_validation_steps = get_generators(
        input_raw_handle, input_label_handle,
        batch_size=config["batch_size"],
        image_shape=config["image_shape"],
        split_file=config["split_file"],
        train_split_ratio=config["train_split_ratio"],
        dist_transform = True,
        overwrite=overwrite,
        labels = None,
        augment = False,
        augment_flip = True,
        augment_distortion_factor = 0.25,
        permute=False)

    fh = open('report2.txt', 'w')
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    fh.close()
    # plot_model(model, to_file='model.png')

    # TODO: add sample weights

        # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"],
                class_weights=class_weights)

if __name__ == "__main__":
    main(sys.argv[1:])