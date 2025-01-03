
from flask import Flask, jsonify, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


app = Flask(__name__)


file_path = '/Users/esha/downloads/Random_Sales_Dataset.csv'  
df = pd.read_csv(file_path)


df['Date'] = pd.to_datetime(df['Date'])


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
    
    days = int(request.args.get('days', 30))  # Default is 30 days

    
    forecast = model_fit.forecast(steps=days)

   
    print(f"Forecast: {forecast}")

    
    return jsonify({"forecast": forecast.tolist()})

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='127.0.0.1', port=8080, debug=True)
