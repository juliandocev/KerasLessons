import numpy as np
from keras.models import Sequential
from keras.models import load_model

model = load_model('../NeuralNetworks/FoodTypeANN2.h5')

# input data1
input_data = np.empty(shape=7, dtype=float)
input_data[0] = float(input('Enter Fat: '))
input_data[1] = float(input('Enter Protein: '))
input_data[2] = float(input('Enter Carbohydrate: '))
input_data[3] = float(input('Enter Sugars: '))
input_data[4] = float(input('Enter Fiber: '))
input_data[5] = float(input('Enter Cholesterol: '))
input_data[6] = float(input('Enter Saturated Fat: '))
print(input_data.shape)
input_data = input_data.reshape(1, -1)
prediction = model.predict(input_data)


print(prediction)



