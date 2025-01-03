# filepath: /Users/esha/project/app.py
from flask import Flask, jsonify, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Initialize Flask app
app = Flask(__name__)

# Load your sales data
file_path = '/Users/esha/downloads/Random_Sales_Dataset.csv'  # Update with the correct path
df = pd.read_csv(file_path)

# Make sure Date is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Fit ARIMA model
model = ARIMA(df['Sales'], order=(5,1,0))  # Adjust p, d, q based on your dataset
model_fit = model.fit()
print("Model fitting complete")
print("Model summary:")
print(model_fit.summary())

@app.route('/')
def home():
    return "Welcome to the Sales Forecasting API!"

@app.route('/predict', methods=['GET'])
def predict():
    # Get the number of days for the forecast from the user
    days = int(request.args.get('days', 30))  # Default is 30 days

    # Make forecast for the given number of days
    forecast = model_fit.forecast(steps=days)

    # Debugging print to see if forecast is generated
    print(f"Forecast: {forecast}")

    # Return the forecast as a JSON response
    return jsonify({"forecast": forecast.tolist()})

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='127.0.0.1', port=8080, debug=True)