import os
from __myconfig import initconfig
config = initconfig()

from unet.prediction import run_validation_cases


def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(split_keys_file=config["split_file"],
                         model_file=config["model_file"],
                         labels=config["labels"],
                         raw_file=config["data_file"],
                         label_file=config["label_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    main()

# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

import matplotlib.pyplot as plt
im = np.max(data[0],axis=2)
plt.imshow(im[:,:,0])
im = np.max(prediction[0,:,:,:],axis=2)
plt.imshow(im[:,:,0],clim=(0.0, 1))

im = np.max(predictions[21],axis=2)
plt.imshow(im[:,:,0],clim=(0.0, 1))

import SimpleITK as sitk

i1 = sitk.GetImageFromArray(np.swapaxes(predictions[21][:,:,:,1], 2, 0))
sitk.Show(i1)