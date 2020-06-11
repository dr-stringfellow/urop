import numpy as np
import os
import keras
from keras.layers import Input, Dropout
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
import keras.backend as K
import tensorflow as tf

def huber(y_true, y_pred):
  diff = y_true - y_pred
  sq = 0.5 * K.square(diff)
  lin  = K.abs(diff) - 0.5
  pwise  = K.abs(diff) < 1
  return tf.where(pwise, sq, lin)

trainX = np.load("data/WJets_x_0.npy")
trainY = np.load("data/WJets_y_0.npy")

valX = np.load("data/WJets_x_1.npy")
valY = np.load("data/WJets_y_1.npy")

nn = Sequential()
nn.add(BatchNormalization(input_shape=(9000,6),momentum=0.6))
nn.add(Dense(input_shape=(9000,6),output_dim=20,activation='relu'))
nn.add(BatchNormalization(input_shape=(9000,20),momentum=0.4))
nn.add(Dense(input_shape=(9000,20),output_dim=5,activation='relu'))
nn.add(BatchNormalization(input_shape=(9000,5),momentum=0.3))
nn.add(Dense(input_shape=(9000,5),output_dim=4,activation='relu'))
nn.add(BatchNormalization(input_shape=(9000,4),momentum=0.6))
nn.add(Dense(activation='sigmoid', kernel_initializer='lecun_uniform',input_shape=(9000,4),output_dim=1))
nn.compile(optimizer='Adam', loss=huber)
nn.summary()

nn.fit(trainX, trainY, batch_size=50, epochs=25, validation_data=(valX, valY))

for i in range(9):
    testX = np.load("data/WJets_x_1%i.npy"%i)
    testY = nn.predict(testX)
    np.save('data/WJets_pred_1%i.npy'%i, testY)

