import pickle
import os
import collections
import socket
import h5py


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def setup_paths():
    # setup paths/environments
    # set the path based on machine
    if socket.gethostname() == 'base-ws1':
        datafold = '/nrs/mouselight/Users/base/annotated_neuron'

    elif socket.gethostname() == 'vega':
        # do nothing
        datafold = '/nrs/mouselight/Users/base/annotated_neuron'
    else:
        # do nothing
        datafold = '/nrs/mouselight/Users/base/annotated_neuron'
    return datafold

def preload_data(input_h5_file='2017-09-25_G-007_consensus-training_raw.h5:volume'):
    # load data
    datafold = setup_paths()
    # check for dataset
    file,dataset = input_h5_file.split(':')
    input_raw_f = os.path.join(datafold, file).replace('/', '//')
    input_raw_handle = h5py.File(input_raw_f, 'r')[dataset]
    return input_raw_handle