from keras import backend as K
from unet.training import load_old_model
import numpy as np
model_file = '/groups/mousebrainmicro/home/base/CODE/MOUSELIGHT/unetAxon/pipe3D/axon_segmentation_model_20180801-135027.h5'
model = load_old_model(model_file)

image_size = (48,48,48,1)
print(model.summary())

input_img = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_name = 'conv3d_1'
filter_index = 0  # can be any integer from 0 to 32, as there are 32 filters in that layer

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, :, filter_index])
# for filter_index in range(layer_output.shape[-1]):
#     loss = K.mean(layer_output[:, :, :, :, filter_index])
#     print(loss)

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])
# step size for gradient ascent
step = 1.

# we start from a gray image with some noise
input_img_data = np.random.random((1, image_size[0], image_size[1], image_size[2], 2)) * 3000 + 12000.
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

import SimpleITK as sitk
i1 = sitk.GetImageFromArray(np.swapaxes(np.asarray(input_img_data[0,...,0],np.uint16),2,0))
sitk.Show(i1)
