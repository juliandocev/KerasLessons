import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


#data = pd.read_excel('../TestData/FoodDatabase minimalistic.xlsx', sheet_name='SR Legacy and FNDDS')
data = pd.read_excel('../TestData/FoodDatabaseBigger.xlsx', sheet_name='SR Legacy and FNDDS')
data.shape

# data.drop('name', axis=0, inplace=True)
del data['Food Group']
del data['name']
del data['ID']

array = data.to_numpy()

#input = array[:, [1, 2, 3]]
input = array[:, [1, 2, 3, 4, 5, 6, 7]] #vzemam samo purvite 7 stoinosti za test
answers = array[:, 0]  # sled iztrivaneto nulevata kolonka stava food group number
# output = np.empty((0, 22), dtype=float)
# for i in range(len(answers)):
#     row = np.zeros(shape=[22])
#     row[int(answers[i]) - 1] = 1
#     output = np.append(output, np.array([row]), axis=0)

# Save output
# np.save('../TestData/output', output)
# np.save('../TestData/input', input)
np.save('../TestData/input_bigger', input)
#print(input)



