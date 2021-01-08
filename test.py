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