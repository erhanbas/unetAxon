from __future__ import print_function
import glob
import os

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

#UNCOMMENT BELOW TO DETERMINE GPU
# from keras import backend as K
# import tensorflow as tf
#Use one GPU
# if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # # Automatically choose an existing and supported device if the specified one does not exist
    # config_GPU = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # # To constrain the use of gpu memory, otherwise all memory is used
    # config_GPU.gpu_options.allow_growth = True
    # sess = tf.Session(config=config_GPU)
    # K.set_session(sess)
###############################################################################################




def main():
    overwrite = True
    # training data handle
    input_raw_handle = preload_data(config['data_file'])
    input_label_handle = preload_data(config['label_file'])

    # load model
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = unet3D.generate_unet3D(input_shape=config["input_shape"],
                                   pool_size=config["pool_size"],
                                   n_labels=config["n_labels"],
                                   initial_learning_rate=config["initial_learning_rate"],
                                   deconvolution=config["deconvolution"])
    print(model.summary())

    # generators for training and validation with augmentation

    class_frequencies = np.array([np.prod(input_label_handle.shape)-np.sum(input_label_handle[:]>0),np.sum(input_label_handle[:]>0)])
    class_weights = class_frequencies.sum() / class_frequencies.astype(np.float32)
    # class_weights = class_weights ** 0.5
    print (class_weights)

    train_generator, validation_generator, n_train_steps, n_validation_steps = get_generators(
        input_raw_handle, input_label_handle,
        batch_size=config["batch_size"],
        image_shape=config["image_shape"],
        split_file=config["split_file"],
        train_split_ratio=config["train_split_ratio"],
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