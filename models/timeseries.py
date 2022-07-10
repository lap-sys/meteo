# 0. Install and Import Dependencies

#!pip install neuralprophet

import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import time
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns",None)
# 1. Read in Data and Process Dates
df = pd.read_csv('../weatherAUS.csv')

df.Location.unique()
df.columns
melb = df[df['Location']=='Melbourne']
melb['Date'] = pd.to_datetime(melb['Date'])
melb.head()
plt.plot(melb['Date'], melb['Temp3pm'])
plt.show()
melb['Year'] = melb['Date'].apply(lambda x: x.year)
melb = melb[melb['Year']<=2015]
plt.plot(melb['Date'], melb['Temp3pm'])
plt.show()
data = melb[['Date', 'Temp3pm']] 
data.dropna(inplace=True)
data.columns = ['ds', 'y'] 
data.head()
# 2. Train Model
m = NeuralProphet()
model = m.fit(data, freq='D')#, epochs=1000)
# 3. Forecast Away
future = m.make_future_dataframe(data, periods=900)
forecast = m.predict(future)
forecast.head()
plot1 = m.plot(forecast)
plt2 = m.plot_components(forecast)
# 4. Save Model

with open('timeseries_fbprophet.pkl', "wb") as f:
    pickle.dump(m, f)
del m
####################
with open('timeseries_fbprophet.pkl', "rb") as f:
    m = pickle.load(f)
future = m.make_future_dataframe(data, periods=900)
forecast = m.predict(future)
forecast.head()
plot1 = m.plot(forecast)

