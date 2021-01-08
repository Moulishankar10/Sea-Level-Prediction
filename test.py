# INDIAN CURRENCY VALUE PREDICTION

# DEVELOPED BY: 
# MOULISHANKAR M R

# IMPORTING MODULES
import numpy as np
import pandas as pd
import datetime as dt
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# IMPORTING DATA
data = pd.read_csv("data/data.csv")

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
