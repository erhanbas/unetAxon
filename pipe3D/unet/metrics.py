from functools import partial

from keras import backend as K
import tensorflow as tf
import numpy as np

def segmetric(y_true,y_pred,mask_thr=0):
    """converts input label to normalized distance image
        maps skeleton to 1, radius to 0

    """
    input_mask = np.greater(y_true,mask_thr)
    #skeletonize input mask
    out = []
    for ind in range(input_mask.shape[0]):
        input_data = np.asarray(input_mask[ind,...,0],np.float)
        dist2boundary = distance_transform_edt(input_data)
        out.append(sigmoid(dist2boundary-1)) # at boundary => 0.5

    return np.asarray(out)[...,None] # extend

def detection_metric(y_true, y_pred, thr = 1.0,smooth=1.0):
    y_true_f = K.flatten(K.less_equal(y_true,thr))
    y_pred_f = K.flatten(K.less_equal(y_pred,thr))
    y_true_bf = tf.cast(y_true_f,tf.float32)
    y_pred_bf = tf.cast(y_pred_f,tf.float32)
    intersection = K.sum(y_true_bf * y_pred_bf)
    return (2. * intersection + smooth) / (K.sum(y_true_bf) + K.sum(y_pred_bf) + smooth)

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))
def mask_mean_squared_error_loss(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weight = K.less_equal(y_true_f, 15)
    loss = tf.losses.mean_squared_error(y_pred_f, y_true_f,weight,reduction="weighted_sum_over_batch_size")
    return loss

def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)

def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])

def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
