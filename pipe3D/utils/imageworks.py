import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d, medial_axis
from scipy.ndimage.morphology import distance_transform_edt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def image2dist(inputimage,target_label = 1, mask_thr = 0):
    """converts input label to normalized distance image
        maps skeleton to 0
    """
    input_mask = np.greater(inputimage==target_label,mask_thr)
    out = []
    for ind in range(input_mask.shape[0]):
        input_data = np.asarray(input_mask[ind,...,0],np.float)
        # dist2boundary = distance_transform_edt(input_data)
        # out.append(sigmoid(dist2boundary-1)) # at boundary => 0.5
        dist2center = distance_transform_edt(input_data < 1)
        out.append(dist2center) # at boundary => 0.5

    return np.asarray(out)[...,None] # extend
