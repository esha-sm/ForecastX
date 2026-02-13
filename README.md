# ForecastX – Self-Serve Sales Forecasting Tool
## Live Demo

👉 https://forecastx.streamlit.app/


ForecastX is a lightweight web application that allows users to upload historical sales data and generate forecasts through a guided modeling workflow.

## Why I Built This

I wanted to understand how forecasting tools are structured end-to-end, so I built a web app that handles data cleaning, model fitting, and visualization in one workflow.

## Core Features

- Default dataset  
- Automated preprocessing (date parsing, missing value handling)  
- ARIMA-based time series forecasting  
- Interactive visualizations (Plotly)  
- Forecast horizon customization  
- Exportable predictions  
- error handling  

## Tech Stack

- Python  
- Streamlit  
- Statsmodels (ARIMA)  
- Scikit-learn  
- Plotly  
- Pandas  

## How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py



