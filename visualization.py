import pandas as pd

# Specify the path to the CSV file
file_path = 'Random_Sales_Dataset.csv'

file_path = '/Users/esha/downloads/Random_Sales_Dataset.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the data
print(df.head())

# Display detailed information about the dataframe
print(df.info())

# Display a summary of statistics for the numerical columns
print(df.describe())

# Display the first 10 rows
print(df.head(366))  # Change the number inside the parentheses to display more rows
print(df.columns)
df.columns = df.columns.str.strip()  # This removes leading/trailing spaces from column names

import matplotlib.pyplot as plt

# Convert the Date column to datetime format (if not already done)
df['Date'] = pd.to_datetime(df['Date'])

# Create a line plot for sales trends over time
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Sales'], marker='o', color='blue', linestyle='-', linewidth=2)
plt.title("Sales Trend Over Time", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import seaborn as sns

# Create a histogram for the sales distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Sales'], kde=True, bins=30, color='green')
plt.title("Sales Distribution", fontsize=14)
plt.xlabel("Sales", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

# Create a box plot for sales by product category
plt.figure(figsize=(10, 6))
sns.boxplot(x='Product', y='Sales', data=df, palette="Set2")
plt.title("Sales Distribution by Product Category", fontsize=14)
plt.xlabel("Product", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.tight_layout()
plt.show()

# Create a bar plot for sales by region
plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='Sales', data=df, estimator='sum', palette="Blues_d")
plt.title("Total Sales by Region", fontsize=14)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.tight_layout()
plt.show()

# Create a count plot for the product distribution
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

# Load the dataset
df = pd.read_csv(file_path)


# Ensure that the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Feature engineering
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Day of the week (0=Monday, 6=Sunday)
df['Month'] = df['Date'].dt.month  # Month (1=January, 12=December)
df['Quarter'] = df['Date'].dt.quarter  # Quarter (1=Q1, 4=Q4)
df['Sales_Lag'] = df['Sales'].shift(1)  # Previous day's sales as a lag feature
df['RollingMean'] = df['Sales'].rolling(window=7).mean()  # 7-day moving average of sales

# Display the dataframe to check the new features
print(df.head())

import statsmodels
print(statsmodels.__version__)  # Prints the version of statsmodels

from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model (adjust p, d, q based on your dataset)
model = ARIMA(df['Sales'], order=(5,1,0))  # p, d, q are hyperparameters for ARIMA
model_fit = model.fit()

# Make forecast
forecast = model_fit.forecast(steps=30)  # Forecast next 30 days
print(forecast)

import matplotlib.pyplot as plt

# Plot historical sales
plt.plot(df['Date'], df['Sales'], label='Historical Sales')

# Plot forecasted sales
forecast_dates = pd.date_range(start=df['Date'].iloc[-1], periods=31, freq='D')[1:]  # Get forecast dates
plt.plot(forecast_dates, forecast, label='Forecasted Sales', linestyle='--', color='orange')

plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast')
plt.legend()
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Create feature matrix (X) and target vector (y)
X = df[['DayOfWeek', 'Month', 'Quarter', 'Sales_Lag', 'RollingMean']]
y = df['Sales']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions and evaluate
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
