import pandas as pd

# Specify the path to the CSV file
file_path = 'Random_Sales_Dataset.csv'

file_path = '/Users/esha/downloads/Random_Sales_Dataset.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)


print(df.head())


print(df.info())


print(df.describe())


print(df.head(366)) 
print(df.columns)
df.columns = df.columns.str.strip()  

import matplotlib.pyplot as plt


df['Date'] = pd.to_datetime(df['Date'])


plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Sales'], marker='o', color='blue', linestyle='-', linewidth=2)
plt.title("Sales Trend Over Time", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




plt.figure(figsize=(10, 6))
sns.histplot(df['Sales'], kde=True, bins=30, color='green')
plt.title("Sales Distribution", fontsize=14)
plt.xlabel("Sales", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Product', y='Sales', data=df, palette="Set2")
plt.title("Sales Distribution by Product Category", fontsize=14)
plt.xlabel("Product", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='Sales', data=df, estimator='sum', palette="Blues_d")
plt.title("Total Sales by Region", fontsize=14)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='Product', data=df, palette="Pastel1")
plt.title("Sales Frequency by Product", fontsize=14)
plt.xlabel("Product", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()

import pandas as pd
file_path = 'Random_Sales_Dataset.csv'

file_path = '/Users/esha/downloads/Random_Sales_Dataset.csv'


df = pd.read_csv(file_path)



df['Date'] = pd.to_datetime(df['Date'])


df['DayOfWeek'] = df['Date'].dt.dayofweek  
df['Month'] = df['Date'].dt.month  
df['Quarter'] = df['Date'].dt.quarter  
df['Sales_Lag'] = df['Sales'].shift(1)  
df['RollingMean'] = df['Sales'].rolling(window=7).mean()  


print(df.head())

import statsmodels
print(statsmodels.__version__)  

from statsmodels.tsa.arima.model import ARIMA


model = ARIMA(df['Sales'], order=(5,1,0))  # p, d, q are hyperparameters for ARIMA
model_fit = model.fit()


forecast = model_fit.forecast(steps=30)  
print(forecast)

import matplotlib.pyplot as plt


plt.plot(df['Date'], df['Sales'], label='Historical Sales')


forecast_dates = pd.date_range(start=df['Date'].iloc[-1], periods=31, freq='D')[1:]  
plt.plot(forecast_dates, forecast, label='Forecasted Sales', linestyle='--', color='orange')

plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast')
plt.legend()
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


X = df[['DayOfWeek', 'Month', 'Quarter', 'Sales_Lag', 'RollingMean']]
y = df['Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)



model_fit = model.fit()
print("Model is trained and ready for forecasting")
