import os

import nibabel as nib
import numpy as np
# import tables
import h5py
from .training import load_old_model
from utils.io_utils import pickle_load, pickle_dump, preload_data
from utils.patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices
from .augment import permute_data, generate_permutation_keys, reverse_permute_data
import graphviz

def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, tf_flag=True):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    if tf_flag:
        data_shape = data.shape[1:4]
    else:
        data_shape = data.shape[-3:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > data_shape) * ((patch_index + patch_shape) - data_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index

def patch_wise_prediction(model, data, overlap=0, batch_size=1, permute=False,label_data=None, tf_flag=True):
    """
    :param batch_size:
    :param model:
    :param data[patch/z/y/x/ch]:
    :param overlap:
    :return:
    """
    if tf_flag:
        patch_shape = tuple([int(dim) for dim in model.input.shape[1:4]])
        data_shape = data.shape[1:4]
    else:
        patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
        data_shape = data.shape[-3:]

    predictions = list()
    grid_subs = compute_patch_indices(data_shape, patch_size=patch_shape, overlap=overlap)
    batch = list()
    i = 0
    while i < len(grid_subs):
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(data[0], patch_shape=patch_shape, patch_index=grid_subs[i])
            patch.shape
            batch.append(patch)
            i += 1
        prediction = predict(model, np.asarray(batch), permute=permute)

        import tensorflow as tf
        lab = get_patch_from_3d_data(label_data[227][...,None], patch_shape=patch_shape, patch_index=grid_subs[i])
        y_true = tf.keras.utils.to_categorical(lab, num_classes=2)
        y_pred = prediction

        y_true_f = K.flatten(K.cast_to_floatx(y_true))
        y_pred_f = K.flatten(K.cast_to_floatx(y_pred))
        intersection = K.sum(y_true_f * y_pred_f)
        (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

        from unet.metrics import dice_coefficient
        er=dice_coefficient(K.cast_to_floatx(y_true[None,...]),K.cast_to_floatx(prediction))
        K.eval(er)


        batch_y.shape

        model.evaluate(patch[None,...],batch_y[None,...])

        get_prediction_labels
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = list(data.shape[1:4]) + [int(model.output.shape[-1])]
    # TODO: fix reconstruct_from_patches
    return reconstruct_from_patches(predictions, patch_indices=grid_subs, data_shape=output_shape)


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data).tolist()[1:]:
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays


def get_test_indices(testing_file):
    return pickle_load(testing_file)


def predict_from_data_file(model, open_data_file, index):
    return model.predict(open_data_file.root.data[index])


def predict_and_get_image(model, data, affine):
    return nib.Nifti1Image(model.predict(data)[0, 0], affine)


def predict_from_data_file_and_get_image(model, open_data_file, index):
    return predict_and_get_image(model, open_data_file.root.data[index], open_data_file.root.affine)


def predict_from_data_file_and_write_image(model, open_data_file, index, out_file):
    image = predict_from_data_file_and_get_image(model, open_data_file, index)
    image.to_filename(out_file)


def prediction_to_image(prediction, affine, label_map=False, threshold=0.5, labels=None):
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
    elif prediction.shape[1] > 1:
        if label_map:
            label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction, affine)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    return nib.Nifti1Image(data, affine)


def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(nib.Nifti1Image(prediction[0, i], affine))
    return prediction_images


def run_validation_case(data_index, output_dir, model, raw_data, label_file=None,
                        output_label_map=False, threshold=0.5, labels=None, overlap=16, permute=False):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is 
    considered a positive result and will be assigned a label.  
    :param labels:
    :param training_modalities:
    :param data_file:
    :param model:
    """
    training_modalities = None
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # affine = data_file.root.affine[data_index]
    affine = np.eye(4, 4)
    test_data = np.asarray([raw_data[data_index]])
    test_data = np.transpose(test_data, (0, 2, 3, 4, 1))
    image = test_data[0][...,0]

    if label_file:
        label_data = preload_data(label_file)
        label = label_data[data_index]==1
    else:
        label_data = None

    #model.input.shape[-3:] for th
    patch_shape = tuple([int(dim) for dim in model.input.shape[1:4]])
    if patch_shape == test_data.shape[1:4]:
        prediction = predict(model, test_data, permute=permute,label_data=label_data)
    else:
        raw_shape = test_data.shape[1:4]
        image_shape = patch_shape
        center_vox = np.round((np.asarray(raw_shape[1:-1]) + 1) / 2)

        start_vox = np.asarray(center_vox - np.asarray(image_shape)/2,np.int)
        end_vox = np.asarray(start_vox + image_shape,np.int)
        batch_x = crop_image(test_data,start_vox,end_vox)
        # batch_y = crop_image(label_data,start_vox,end_vox)
        prediction = predict(model, batch_x, permute=permute)
        plt.figure(0)
        plt.imshow(np.max(prediction[0, ..., 0], axis=0))
        # prediction = patch_wise_prediction(model=model, data=test_data, overlap=overlap, permute=permute,label_data=label_data)[np.newaxis]
    prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold,
                                           labels=labels)
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))
    else:
        prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))
    if label_file:
        label_data.close()


def run_validation_cases(split_keys_file, model_file, labels, raw_file, label_file,
                         output_label_map=False, output_dir=".", threshold=0.5, overlap=16, permute=False):
    split_indices = pickle_load(split_keys_file)
    validation_indices = split_indices['test']
    model = load_old_model(model_file)
    raw_data = preload_data(raw_file)
    for index in validation_indices:
        case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
        run_validation_case(data_index=index, output_dir=case_directory, model=model, raw_data=raw_data,
                            label_file=label_file, output_label_map=output_label_map, labels=labels,
                            threshold=threshold, overlap=overlap, permute=permute)

    raw_data.close()

def crop_image(input_image,start_vox,end_vox):
    return input_image[:, start_vox[0]:end_vox[0], start_vox[1]:end_vox[1], start_vox[2]:end_vox[2],:]

def predict(model, data, permute=False):
    if permute:
        predictions = list()
        for batch_index in range(data.shape[0]):
            predictions.append(predict_with_permutations(model, data[batch_index]))
        return np.asarray(predictions)
    else:
        return model.predict(data)


def predict_with_permutations(model, data):
    predictions = list()
    for permutation_key in generate_permutation_keys():
        temp_data = permute_data(data, permutation_key)[np.newaxis]
        predictions.append(reverse_permute_data(model.predict(temp_data)[0], permutation_key))
    return np.mean(predictions, axis=0)
