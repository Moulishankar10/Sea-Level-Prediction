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

# LOADING THE TRAINED MODEL
model = load_model("model/model",custom_objects=None,compile=True)

# INPUT DATA
print("\nEnter the Time Period on when you want to explore the prediction !")
input_month = input("\nTime Period (MM-YYYY) : ")
month = ["January","February","March","April","May","June","July","August","September","October","November","December"]

# PREPROCESSING INPUT DATA
x_str = dt.datetime(int(input_month[-4:]),int(input_month[:2]),1)
x_pred = (x_str.year - initial.year) * 12 + (x_str.month - initial.month)
x_pred = np.array(x_pred)
x_pred = np.reshape(x_pred, (-1,1))

# SCALING INPUT DATA
xpred_scaled = scaler_x.transform(x_pred)

# PREDICTING THE RESULTANT VALUE
ypred_scaled = model.predict(xpred_scaled)
y_pred = scaler_y.inverse_transform(ypred_scaled)

# DISPLAYING THE RESULTS
print(f"\n\n As per the prediction, by {month[int(input_month[:2])-1]} {int(input_month[-4:])}, the Global Mean Sea Level might be changed to {round(float(y_pred),1)} mm (Monthly Average)\n\n")
