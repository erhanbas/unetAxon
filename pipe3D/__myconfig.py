import os
import glob
import time

today = time.strftime('%Y%m%d-%H%M%S')

def initconfig():
    config = dict()
    config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
    config["image_shape"] = (32, 32, 32)  # This determines what shape the images will be cropped/resampled to.
    config["patch_shape"] = None  # switch to None to train on the whole image
    config["labels"] = (0, 1)  # the label numbers on the input image
    config["n_labels"] = len(config["labels"])
    # config["all_modalities"] = ["t1", "t1Gd", "flair", "t2"]
    # config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
    config["nb_channels"] = 2
    if "patch_shape" in config and config["patch_shape"] is not None:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
    else:
        config["input_shape"] = tuple(list(config["image_shape"])+[config["nb_channels"]])

    # config["truth_channel"] = config["nb_channels"]
    config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

    config["batch_size"] = 6
    config["validation_batch_size"] = 12
    config["n_epochs"] = 500  # cutoff the training after this many epochs
    config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
    config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
    config["initial_learning_rate"] = 0.00001
    config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
    config["train_split_ratio"] = 0.8  # portion of the data that will be used for testing
    config["flip"] = False  # augments the data by randomly flipping an axis during
    config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
    config["distort"] = None  # switch to None if you want no distortion
    config["augment"] = config["flip"] or config["distort"]
    config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
    config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
    config["skip_blank"] = True  # if True, then patches without any target will be skipped

    config["data_file"] = "2017-09-25_G-007_consensus-training_raw.h5:volume"
    config["label_file"] = "2017-09-25_G-007_consensus-training_dense_label.h5:volume"
    config["model_file"] = os.path.abspath('axon_segmentation_model_{}.h5'.format(today))
    config["split_file"] = os.path.abspath("split_ids.pkl")
    # config["training_file"] = os.path.abspath("training_ids.pkl")
    # config["validation_file"] = os.path.abspath("validation_ids.pkl")
    config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
    return config

def initconfig_old():
    config = dict()
    config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
    config["image_shape"] = (144, 144, 144)  # This determines what shape the images will be cropped/resampled to.
    config["patch_shape"] = (64, 64, 64)  # switch to None to train on the whole image
    config["labels"] = (1, 2, 4)  # the label numbers on the input image
    config["n_labels"] = len(config["labels"])
    config["all_modalities"] = ["t1", "t1Gd", "flair", "t2"]
    config["training_modalities"] = config[
        "all_modalities"]  # change this if you want to only use some of the modalities
    config["nb_channels"] = len(config["training_modalities"])
    if "patch_shape" in config and config["patch_shape"] is not None:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
    else:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
    config["truth_channel"] = config["nb_channels"]
    config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

    config["batch_size"] = 6
    config["validation_batch_size"] = 12
    config["n_epochs"] = 500  # cutoff the training after this many epochs
    config[
        "patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
    config[
        "early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
    config["initial_learning_rate"] = 0.00001
    config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
    config["validation_split"] = 0.8  # portion of the data that will be used for training
    config["flip"] = False  # augments the data by randomly flipping an axis during
    config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
    config["distort"] = None  # switch to None if you want no distortion
    config["augment"] = config["flip"] or config["distort"]
    config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
    config["training_patch_start_offset"] = (
    16, 16, 16)  # randomly offset the first patch index by up to this offset
    config["skip_blank"] = True  # if True, then patches without any target will be skipped

    config["data_file"] = os.path.abspath("brats_data.h5")
    config["model_file"] = os.path.abspath("tumor_segmentation_model.h5")
    config["training_file"] = os.path.abspath("training_ids.pkl")
    config["validation_file"] = os.path.abspath("validation_ids.pkl")
    config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
    return config