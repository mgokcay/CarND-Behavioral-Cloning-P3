from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

# Simple implementation of this conv net is inspired by the NVIDIA network described in
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# I additionally use cropping and dopouts in the fully connecged layers.


def NvidiaNet():
    model = Sequential()

    # crops at top and bottom, output shape = (75, 320, 3)
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    # shape = (65, 320, 3)
    # normalization
    model.add(Lambda(lambda x: (x / 255) - 0.5))
    # shape = (65, 320, 3)

    # convolutional layers
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))

    # flattening
    model.add(Flatten())

    # fully connected layers with dropouts
    model.add(Dense(100))
    model.add(Dropout(0.4))
    model.add(Dense(50))
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    return model