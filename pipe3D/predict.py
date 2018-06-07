import os
from __myconfig import initconfig
config = initconfig()

from unet.prediction import run_validation_cases


def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(split_keys_file=config["split_file"],
                         model_file=config["model_file"],
                         training_modalities=None,
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    main()