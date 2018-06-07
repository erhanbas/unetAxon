import numpy as np
from keras import backend as K
# K.set_image_dim_ordering('tf')
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.optimizers import Adam

from unet.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
# K.set_image_dim_ordering('tf')
K.set_image_data_format("channels_last")
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate
# from keras.models import Model
# from keras.layers import Input,  concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization, \
#     Conv3D, Conv3DTranspose, MaxPooling3D


def generate_unet3D(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                     depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                     batch_normalization=False, activation_name="sigmoid"):
    """
    Based on github repo of Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    :param base_num_filters:
    :param num_classes:
    :param kernel_size:
    :param image_width:
    :param image_height:
    :return:
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    # left half of U
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    # right half of U wit concatenate from left
    # for th: current_layer._keras_shape[1] to get the feature
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                            n_filters=current_layer._keras_shape[-1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=-1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    return model

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)

# def generate_3D_unet_old(base_num_filters, num_classes, kernel_size=(3,3,3), image_width=128, image_height=128, image_depth=128):
#         """
#         :param base_num_filters:
#         :param num_classes:
#         :param kernel_size:
#         :param image_width:
#         :param image_height:
#         :return:
#         """
#         layer_depth = 4
#
#
#
#         inputs = Input((image_height,  image_width, image_depth, 1))
#         conv1 = Conv3D(base_num_filters, kernel_size, activation='relu', padding='same')(inputs)
#         conv1 = Conv3D(base_num_filters, kernel_size, activation='relu', padding='same')(conv1)
#         pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
#
#         conv2 = Conv3D(2*base_num_filters, kernel_size, activation='relu', padding='same')(pool1)
#         conv2 = Conv3D(2*base_num_filters, kernel_size, activation='relu', padding='same')(conv2)
#         pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
#
#         conv3 = Conv3D(4*base_num_filters, kernel_size, activation='relu', padding='same')(pool2)
#         conv3 = Conv3D(4*base_num_filters, kernel_size, activation='relu', padding='same')(conv3)
#         pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
#
#         conv4 = Conv3D(8*base_num_filters, kernel_size, activation='relu', padding='same')(pool3)
#         conv4 = Conv3D(8*base_num_filters, kernel_size, activation='relu', padding='same')(conv4)
#         pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
#
#         conv5 = Conv3D(16*base_num_filters, kernel_size, activation='relu', padding='same')(pool4)
#         conv5 = Conv3D(16*base_num_filters, kernel_size, activation='relu', padding='same')(conv5)
#         drop = Dropout(0.5)(conv5)
#
#         up6 = concatenate([Conv3DTranspose(8*base_num_filters, kernel_size, strides=(2, 2, 2), padding='same')(drop), conv4], axis=4)
#         conv6 = Conv3D(8*base_num_filters, kernel_size, activation='relu', padding='same')(up6)
#         conv6 = Conv3D(8*base_num_filters, kernel_size, activation='relu', padding='same')(conv6)
#
#         up7 = concatenate([Conv3DTranspose(4*base_num_filters, kernel_size, strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
#         conv7 = Conv3D(4*base_num_filters, kernel_size, activation='relu', padding='same')(up7)
#         conv7 = Conv3D(4*base_num_filters, kernel_size, activation='relu', padding='same')(conv7)
#
#         up8 = concatenate([Conv3DTranspose(2*base_num_filters, kernel_size, strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
#         conv8 = Conv3D(2*base_num_filters, kernel_size, activation='relu', padding='same')(up8)
#         conv8 = Conv3D(2*base_num_filters, kernel_size, activation='relu', padding='same')(conv8)
#
#         up9 = concatenate([Conv3DTranspose(base_num_filters, kernel_size, strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
#         conv9 = Conv3D(base_num_filters, kernel_size, activation='relu', padding='same')(up9)
#         conv9 = Conv3D(base_num_filters, kernel_size, activation='relu', padding='same')(conv9)
#
#         conv10 = Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv9)
#
#         model = Model(inputs=[inputs], outputs=[conv10])
#
#         return model
# def generate_unet(base_num_filters, num_classes, kernel_size=(3,3), image_width=128, image_height=128):
#         """
#         Simple UNet without batch normalization
#         """
#         inputs = Input((image_height,  image_width, 1))
#         conv1 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(inputs)
#         conv1 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(conv1)
#         pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#         conv2 = Conv2D(2*base_num_filters, kernel_size, activation='relu', padding='same')(pool1)
#         conv2 = Conv2D(2*base_num_filters, kernel_size, activation='relu', padding='same')(conv2)
#         pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#         conv3 = Conv2D(4*base_num_filters, kernel_size, activation='relu', padding='same')(pool2)
#         conv3 = Conv2D(4*base_num_filters, kernel_size, activation='relu', padding='same')(conv3)
#         pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#         conv4 = Conv2D(8*base_num_filters, kernel_size, activation='relu', padding='same')(pool3)
#         conv4 = Conv2D(8*base_num_filters, kernel_size, activation='relu', padding='same')(conv4)
#         pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#         conv5 = Conv2D(16*base_num_filters, kernel_size, activation='relu', padding='same')(pool4)
#         conv5 = Conv2D(16*base_num_filters, kernel_size, activation='relu', padding='same')(conv5)
#         drop = Dropout(0.5)(conv5)
#
#         up6 = concatenate([Conv2DTranspose(8*base_num_filters, kernel_size, strides=(2, 2), padding='same')(drop), conv4], axis=3)
#         conv6 = Conv2D(8*base_num_filters, kernel_size, activation='relu', padding='same')(up6)
#         conv6 = Conv2D(8*base_num_filters, kernel_size, activation='relu', padding='same')(conv6)
#
#         up7 = concatenate([Conv2DTranspose(4*base_num_filters, kernel_size, strides=(2, 2), padding='same')(conv6), conv3], axis=3)
#         conv7 = Conv2D(4*base_num_filters, kernel_size, activation='relu', padding='same')(up7)
#         conv7 = Conv2D(4*base_num_filters, kernel_size, activation='relu', padding='same')(conv7)
#
#         up8 = concatenate([Conv2DTranspose(2*base_num_filters, kernel_size, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
#         conv8 = Conv2D(2*base_num_filters, kernel_size, activation='relu', padding='same')(up8)
#         conv8 = Conv2D(2*base_num_filters, kernel_size, activation='relu', padding='same')(conv8)
#
#         up9 = concatenate([Conv2DTranspose(base_num_filters, kernel_size, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
#         conv9 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(up9)
#         conv9 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(conv9)
#
#         conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
#
#         model = Model(inputs=[inputs], outputs=[conv10])
#
#         return model
# def generate_batch_norm_unet(base_num_filters, num_classes, kernel_size=(3,3), image_width=128, image_height=128):
#         """
#         UNet with batch normalization
#         """
#         inputs = Input((image_height, image_width, 1))
#         conv1 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(inputs)
#         bn1 = BatchNormalization()(conv1)
#         conv1 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(bn1)
#         bn1 = BatchNormalization()(conv1)
#         pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
#
#         conv2 = Conv2D(2 * base_num_filters, kernel_size, activation='relu', padding='same')(pool1)
#         bn2 = BatchNormalization()(conv2)
#         conv2 = Conv2D(2 * base_num_filters, kernel_size, activation='relu', padding='same')(bn2)
#         bn2 = BatchNormalization()(conv2)
#         pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
#
#         conv3 = Conv2D(4 * base_num_filters, kernel_size, activation='relu', padding='same')(pool2)
#         bn3 = BatchNormalization()(conv3)
#         conv3 = Conv2D(4 * base_num_filters, kernel_size, activation='relu', padding='same')(bn3)
#         bn3 = BatchNormalization()(conv3)
#         pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
#
#         conv4 = Conv2D(8 * base_num_filters, kernel_size, activation='relu', padding='same')(pool3)
#         bn4 = BatchNormalization()(conv4)
#         conv4 = Conv2D(8 * base_num_filters, kernel_size, activation='relu', padding='same')(bn4)
#         bn4 = BatchNormalization()(conv4)
#         pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)
#
#         conv5 = Conv2D(16 * base_num_filters, kernel_size, activation='relu', padding='same')(pool4)
#         bn5 = BatchNormalization()(conv5)
#         conv5 = Conv2D(16 * base_num_filters, kernel_size, activation='relu', padding='same')(bn5)
#         bn5 = BatchNormalization()(conv5)
#         drop = Dropout(0.5)(bn5)
#
#         up6 = concatenate([Conv2DTranspose(8*base_num_filters, kernel_size, strides=(2, 2), padding='same')(drop), conv4], axis=3)
#         conv6 = Conv2D(8 * base_num_filters, kernel_size, activation='relu', padding='same')(up6)
#         bn6 = BatchNormalization()(conv6)
#         conv6 = Conv2D(8 * base_num_filters, kernel_size, activation='relu', padding='same')(bn6)
#         bn6 = BatchNormalization()(conv6)
#
#         up7 = concatenate([Conv2DTranspose(4*base_num_filters, kernel_size, strides=(2, 2), padding='same')(bn6), conv3], axis=3)
#         conv7 = Conv2D(4 * base_num_filters, kernel_size, activation='relu', padding='same')(up7)
#         bn7 = BatchNormalization()(conv7)
#         conv7 = Conv2D(4 * base_num_filters, kernel_size, activation='relu', padding='same')(bn7)
#         bn7 = BatchNormalization()(conv7)
#
#         up8 = concatenate([Conv2DTranspose(2*base_num_filters, kernel_size, strides=(2, 2), padding='same')(bn7), conv2], axis=3)
#         conv8 = Conv2D(2 * base_num_filters, kernel_size, activation='relu', padding='same')(up8)
#         bn8 = BatchNormalization()(conv8)
#         conv8 = Conv2D(2 * base_num_filters, kernel_size, activation='relu', padding='same')(bn8)
#         bn8 = BatchNormalization()(conv8)
#
#         up9 = concatenate([Conv2DTranspose(base_num_filters, kernel_size, strides=(2, 2), padding='same')(bn8), conv1], axis=3)
#         conv9 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(up9)
#         bn9 = BatchNormalization()(conv9)
#         conv9 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(bn9)
#         bn9 = BatchNormalization()(conv9)
#
#         conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(bn9)
#
#         model = Model(inputs=[inputs], outputs=[conv10])
#
#         return model
