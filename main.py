import csv
import cv2
import numpy as np
from nvidianet import NvidiaNet
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense

# Data augmentation
DO_FLIP = True
USE_SIDE_CAMERAS = True
CORRECTION = 0.2


# hyper parameters
EPOCHS = 3

def main():

    lines = []
    with open('./BehavioralClonningData/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []

    for line in lines:
        addSample(images, measurements, line[0], float(line[3]))
        if (USE_SIDE_CAMERAS):
            addSample(images, measurements, line[1], float(line[3]) + CORRECTION)
            addSample(images, measurements, line[2], float(line[3]) - CORRECTION)

    X_train = np.array(images)
    Y_train = np.array(measurements)

    print("Number of samples " + str(len(X_train)))

    model = NvidiaNet()
    model.compile(loss='mse', optimizer='adam')

    history_object = model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=3, verbose=1)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    # save the model
    model.save('model.h5')
    print('Model saved!')

def addSample(images, measurements, source_path, measurement):

    filename = source_path.split('/')[-1]
    current_path = './BehavioralClonningData/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurements.append(measurement)
    if (DO_FLIP):
        images.append(cv2.flip(image, 1))
        measurements.append(-measurement)

if __name__ == '__main__':
    main()