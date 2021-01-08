# SEA LEVEL PREDICTION

# DEVELOPED BY: 
# MOULISHANKAR M R
# VIGNESHWAR RAVICHANDAR

# IMPORTING MODULES
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import save_model 

# IMPORTING DATA
data = pd.read_csv("data/sea-data.csv")

# PREPROCESSING DATA
x = []
initial_str = data["Time"][0]
initial = dt.datetime(int(initial_str[-4:]),int(initial_str[:2]),1)

for i in range(len(data["Time"])):
    final_str = data["Time"][i]
    final = dt.datetime(int(final_str[-4:]),int(final_str[:2]),1)
    diff = (final.year - initial.year) * 12 + (final.month - initial.month)
    x.append(diff)

y = data["GMSL"].values

# SPLITTING THE TRAINING AND VALIDATION DATA
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# RESHAPING THE DATA
x_train = np.reshape(x_train, (-1,1))
x_val = np.reshape(x_val, (-1,1))
y_train = np.reshape(y_train, (-1,1))
y_val = np.reshape(y_val, (-1,1))

# SCALING THE DATA
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

xtrain_scaled = scaler_x.fit_transform(x_train)
ytrain_scaled = scaler_y.fit_transform(y_train)
xval_scaled = scaler_x.fit_transform(x_val)

# BUILDING THE NEURAL NETWORK
model = Sequential()
model.add(Dense(2, input_dim = 1, activation = 'relu', kernel_initializer='normal'))
model.add(Dense(70, activation = 'relu'))
model.add(Dense(70, activation = 'relu'))
model.add(Dense(70, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

# TRAINING THE DATA
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse', 'mae', 'accuracy'])
history = model.fit(xtrain_scaled, ytrain_scaled, epochs = 100, batch_size = 70, validation_split = 0.1, verbose = 1)
print("\n\n ----- Model is trained successfully ! ----- \n\n")

# VISUALISING THE MODEL LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MODEL LOSS')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# ACCURACY OF THE MODEL
ypred_scaled = model.predict(xval_scaled)
y_pred = scaler_y.inverse_transform(ypred_scaled)
acc_score = r2_score(y_val, y_pred)
print(f"\nAccuracy of the model : {round(acc_score*100,2)}%")

# SAVING THE TRAINED MODEL
PATH = './model/model'
save_model(model,PATH)
print(f"\n\n ---- Successfully stored the trained model at {PATH} ---- \n\n")