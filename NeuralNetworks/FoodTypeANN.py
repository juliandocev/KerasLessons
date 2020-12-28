import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib

# input_data = np.load('../TestData/input.npy')
# output = np.load('../TestData/output.npy')
input_data = np.load('../TestData/input_bigger.npy')
output = np.load('../TestData/output.npy')

print(input_data)
print(output)

print(input_data.shape)
print(output.shape)

# layers
model = Sequential()
model.add(Dense(63, activation='relu', input_dim=7))  # 1st hidden
model.add(Dense(63, activation='relu'))  # 2st hidden
model.add(Dense(63, activation='relu'))  # 3rd hidden
model.add(Dense(63, activation='relu'))  # 4th hidden
model.add(Dense(63, activation='relu'))  # 5th hidden
model.add(Dense(63, activation='relu'))  # 6th hidden
model.add(Dense(63, activation='relu'))  # 7th hidden
model.add(Dense(63, activation='relu'))  # 8th hidden
model.add(Dense(63, activation='relu'))  # 9th hidden
model.add(Dense(63, activation='relu'))  # 10th hidden

model.add(Dense(22, activation='softmax'))

# Optimizers
sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.1, nesterov=True)
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-06, amsgrad=True)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(input_data, output, epochs=1000, batch_size=128)
# model.summary()

# model.save('FoodTypeANN.h5')
# model.save('FoodTypeANN2.h5')
# print(device_lib.list_local_devices())
# import tensorflow as tf
#
# print(tf.test.is_built_with_cuda())
# print(tf.config.list_physical_devices('GPU'))
# print(device_lib.list_local_devices())
# exit()
print ("GPUis", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILBLE")
