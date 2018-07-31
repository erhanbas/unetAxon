import matplotlib.pyplot as plt
from __future__ import print_function
import glob
import os

from __myconfig import initconfig
from unet.model import unet3D
from unet.generator import get_generators
from unet.training import load_old_model, train_model
from utils.io_utils import preload_data
from keras.utils import plot_model
import numpy as np

config = initconfig()

def preprocessinput():
    input_raw_handle = preload_data(config['data_file'])
    input_label_handle = preload_data(config['label_file'])

    # load model
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = unet3D.generate_unet3D(input_shape=config["input_shape"],
                                   pool_size=config["pool_size"],
                                   n_labels=config["n_labels"],
                                   initial_learning_rate=config["initial_learning_rate"],
                                   deconvolution=config["deconvolution"])












test_data = data
test_data = test_data[:,26-16:26+16,26-16:26+16,26-16:26+16,:]

imgplot = plt.imshow(np.max(test_data,axis=0))
prediction = predict(model, test_data, permute=permute)

imgplot = plt.imshow(np.max(prediction[0,...,0],axis=0))
