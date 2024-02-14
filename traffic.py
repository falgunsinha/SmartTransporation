

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('./Traffic_data_monitoring.csv')

df['Time'] = pd.to_datetime(df['Time'])
df['Time'] = df['Time'].dt.hour * 60 + df['Time'].dt.minute + df['Time'].dt.second/60

le1 = LabelEncoder()
df['Traffic Situation'] = le1.fit_transform(df['Traffic Situation'])

le2 = LabelEncoder()
df['Day of the week'] = le2.fit_transform(df['Day of the week'])

# Additional Feature Engineering
vehicle_columns = ['CarCount', 'BusCount', 'BikeCount', 'TruckCount']
df['AvgVehicleCount'] = df[vehicle_columns].mean(axis=1)
df['VehicleCountMean'] = df[vehicle_columns].mean(axis=1)
df['VehicleCountStd'] = df[vehicle_columns].std(axis=1)

scaler = StandardScaler()
df[vehicle_columns] = scaler.fit_transform(df[vehicle_columns])

X = df[['Time', 'Day of the week', 'CarCount', 'BusCount', 'Total', 'BikeCount', 'TruckCount', 'AvgVehicleCount', 'VehicleCountMean', 'VehicleCountStd']]
y = df['Traffic Situation']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Decision Tree Model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_rmse = mean_squared_error(y_test, dt_pred, squared=False)

# Gradient Boosting Model (similar to PSO)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_rmse = mean_squared_error(y_test, gb_pred, squared=False)

# RNN Model
X_train_rnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

rnn_model = Sequential()
rnn_model.add(SimpleRNN(64, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
rnn_model.add(Dense(1))
rnn_model.compile(loss='mse', optimizer='adam')
rnn_model.fit(X_train_rnn, y_train, epochs=100, batch_size=32, verbose=0)
rnn_pred = rnn_model.predict(X_test_rnn).flatten()
rnn_rmse = mean_squared_error(y_test, rnn_pred, squared=False)

# Comparison of ML models
plt.figure(figsize=(8, 6))
plt.bar(['Decision Tree', 'Gradient Boosting', 'RNN'], [dt_rmse, gb_rmse, rnn_rmse], color=['blue', 'orange', 'green'])
plt.title('Model Comparison')
plt.ylabel('RMSE')
plt.xlabel('ML Models')
plt.ylim(0, max(dt_rmse, gb_rmse, rnn_rmse) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()