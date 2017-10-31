#making model...

import sys
import os
import numpy as np
import gc
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils


class TrainModel :
    def __init__(self):
        self.nb_classes = 2
        x_train ,x_test, y_train, y_test = np.load("./temp/{0}.npy".format(args[1]))
        self.x_train = x_train.astype("float")
        self.x_test = x_test.astype("float")
        self.y_train = y_train.astype("float")
        self.y_test = y_test.astype("float")

    def train(self, input = None):
        model = Sequential()
        model.add(Dense(50, input_dim = 400, activation = 'relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        #compile model
        model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        if input == None:
            model.fit(self.x_train, self.y_train, batch_size = 50, nb_epoch = 20)
            hdf5 = "./temp/movie-model.hdf5"
            model.save_weights(hdf5)

            score = model.evaluate(self.x_test, self.y_test)
            print ('loss = ' , score[0])
            print('accuracy = ' , score[1])
        return model

if __name__ == "__main__":
  args = sys.argv
  train = TrainModel()
  train.train()
  gc.collect()
