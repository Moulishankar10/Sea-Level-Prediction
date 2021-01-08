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
data = pd.read_csv("sea-data.csv")

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



