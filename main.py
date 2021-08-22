import numpy as np
import matplotlib.pylot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Dense, Dropout, LSTM

#Load Data
company = 'RELIANCE'

start = dt.datetime(2012,1,1)
end = dt.datetime(2021,6,29)

data = web.DataReader(company, 'RELIANCE', start, end)

#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

protection_days = 90

x_train = []
y_train = []

for x in range(protection_days, len(scaled_data)):
    x_train.append(scaled_data[x-protection_days:x, 0])
    y_train.append(scaled_data[x ,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,  y_train, epochs=25, batch_size=32)

##

test_start = dt.datetime(2012,1,1)
test_end = dt.datetime(2021,6,29)

test_data = web.DataReader(company, 'RELIANCE', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((datetime['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_datase - len(test_voice - prediction_days:]).value)]
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaled_data.transform(model_inputs)

#Make Predictions

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model[x-prediction_days:x, 0])

x_test = np.array(x-test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prdicted_prices = model.predict(x_test)
protection_days = scaler.inverse_transform(protection_days)

#Plot the Predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted" {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()