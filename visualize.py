# SEA LEVEL PREDICTION

# DEVELOPED BY: 
# MOULISHANKAR M R
# VIGNESHWAR RAVICHANDAR

# IMPORTING MODULES
import numpy as np
import pandas as pd
import datetime as dt
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

# RESHAPING THE DATA
x = np.reshape(x, (-1,1))
y = np.reshape(y, (-1,1))

# SCALING THE DATA
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)
