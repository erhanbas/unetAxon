from keras.utils import to_categorical
def custom_image_generator(generator, directory, seed=None, batch_size=16, target_size=(128, 128),
                           color_mode="grayscale", class_mode=None, isMask=False, num_classes=None):
    """
    Read images from a dirctory batch-size wise
    If images are masks (e.g. 128x128x1) then convert them to multi-label arrays (e.g. 128x128x3) so that they match the output of UNet
    """
    import numpy as np

    # Read from directory (flow_from_directory)
    iterator = generator.flow_from_directory(directory=directory,
                                             target_size=target_size,
                                             color_mode=color_mode,
                                             class_mode=class_mode,
                                             batch_size=batch_size,
                                             seed=seed,
                                             shuffle=True)

    for batch_x in iterator:
        # if image is a mask convert to a multi-label array (binary matrix: 128x128x3)
        if isMask == True:
            batch_x = to_categorical(batch_x, num_classes)
        yield batch_x