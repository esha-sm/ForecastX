ForecastX – Self-Serve Sales Forecasting Tool

ForecastX is a lightweight web application that allows users to upload historical sales data and generate model-driven revenue projections without needing in-house data science expertise.

Why I Built This

Many small businesses rely on manual spreadsheets for forecasting, which limits accuracy and scalability. I built ForecastX to streamline forecasting into a guided workflow that automates preprocessing, model selection, and visualization — turning raw CSV data into actionable projections in minutes.

Core Features

CSV upload or default dataset

Automated preprocessing (date parsing, missing value handling)

ARIMA-based time series forecasting

Interactive visualizations (Plotly)

Forecast horizon customization

Exportable predictions

Robust error handling

Tech Stack

Python
Streamlit
Statsmodels (ARIMA)
Scikit-learn
Plotly
Pandas

How to Run
pip install -r requirements.txt
streamlit run streamlit_app.py

Design Considerations

Built modular components (visualization.py, app.py) for separation of concerns

Focused on usability for non-technical users

Prioritized preprocessing reliability to handle messy real-world data

Designed for extensibility (additional forecasting models can be added easily)

Future Improvements

Add model comparison (Prophet, XGBoost)

Deploy live demo

Integrate automated hyperparameter tuning


