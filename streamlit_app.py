import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import numpy as np

st.title("ForecastX")
st.write("This app forecasts data using the ARIMA model.")
st.write("You can predict values for a minimum of 1 day and a maximum of 365 days.")
st.markdown("<p style='color:red;'>Upload your own file, or if not, no problem! Just use the default dataset to enjoy the app.</p>", unsafe_allow_html=True)

# Initialize data state
data_loaded = False
df = None

# Option to use default dataset or upload a file
use_default = st.checkbox("Use default dataset")

# Handle default dataset
if use_default:
    try:
        df = pd.read_csv('./Random_Sales_Dataset.csv')
        st.success("Default data loaded successfully!")
        data_loaded = True
    except Exception as e:
        st.error(f"An error occurred while loading the default data: {str(e)}")
else:
    # Handle file upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            data_loaded = True
        except Exception as e:
            st.error(f"An error occurred while loading the data: {str(e)}")

# Early exit if no data
if not data_loaded:
    st.warning("Please upload a CSV file or use the default dataset to continue.")
    st.stop()

# Only show the rest of the app if data is loaded
if data_loaded and df is not None:
    # Display the preview
    st.write("Here is a preview of your dataset:")
    st.dataframe(df.head())
    
    # Allow users to select columns for analysis
    columns = df.columns.tolist()
    if len(columns) < 2:
        st.error("The dataset must contain at least two columns.")
        st.stop()

    date_column = st.selectbox("Select the date column", columns)
    value_column = st.selectbox("Select the value column to forecast", columns)

    # Convert date column to datetime
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        if df[date_column].isna().any():
            st.warning(f"Some dates in {date_column} could not be parsed and were set to NaN.")
            df = df.dropna(subset=[date_column])
        st.success(f"Successfully processed dates in {date_column}")
    except Exception as e:
        st.error(f"Error processing date column: {str(e)}. Please ensure the column contains valid dates.")
        st.stop()

    # Ensure value column is numeric
    try:
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        if df[value_column].isna().any():
            st.warning(f"Some values in {value_column} could not be converted to numbers and were set to NaN.")
    except Exception as e:
        st.error(f"Error converting {value_column} to numeric values: {str(e)}")
        st.stop()

    # Date range selection
    min_date = df[date_column].min().date()
    max_date = df[date_column].max().date()
    start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("End date must be after start date")
        st.stop()

    # Filter data by date range
    df = df[(df[date_column].dt.date >= start_date) & (df[date_column].dt.date <= end_date)]

    # Handle missing values
    missing_value_option = st.selectbox(
        "How would you like to handle missing values?",
        ("Drop rows with missing values", "Fill with mean", "Fill with median")
    )

    if missing_value_option == "Drop rows with missing values":
        df = df.dropna(subset=[value_column])
    elif missing_value_option == "Fill with mean":
        df[value_column] = df[value_column].fillna(df[value_column].mean())
    else:  # Fill with median
        df[value_column] = df[value_column].fillna(df[value_column].median())

    # Check if enough data is available
    if len(df) < 30:
        st.error("Not enough data points (minimum 30 required). Please adjust your date range or upload more data.")
        st.stop()

    # Sort data by date
    df = df.sort_values(by=date_column)

    # Set date column as index
    df.set_index(date_column, inplace=True)

    # ARIMA model fitting
    try:
        # Only use the numeric values for ARIMA
        model = ARIMA(df[value_column], order=(5,1,0))
        model_fit = model.fit()
        st.success("ARIMA model fitted successfully!")
    except Exception as e:
        st.error(f"Error fitting ARIMA model: {str(e)}")
        st.stop()

    # Display date range
    st.markdown(f"<p style='font-size:20px; color:blue;'>Historical data range: {df.index.min().date()} to {df.index.max().date()}</p>", unsafe_allow_html=True)

    # Forecast section
    days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

    if st.button("Generate Forecast"):
        try:
            forecast = model_fit.forecast(steps=days)
            last_date = df.index.max()
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast
            })
            forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
            
            st.write(f"Forecasted values for the next {days} days")
            
            def color_code(val):
                return 'background-color: lightgreen' if val > forecast_df['Forecast'].mean() else 'background-color: lightcoral'
            
            st.dataframe(forecast_df.style.applymap(color_code, subset=['Forecast']))
            
            # Create visualization
            fig = px.line(title='Forecast Results')
            fig.add_scatter(x=df.index, y=df[value_column], name='Historical Data', line=dict(color='blue'))
            fig.add_scatter(x=forecast_dates, y=forecast, name='Forecast', line=dict(color='red'))
            fig.update_layout(xaxis_title='Date', yaxis_title='Value', template='plotly_white')
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")

    # Additional visualizations
    st.markdown("---")
    st.header("Additional Visualizations")

    try:
        st.markdown("**Historical Data Trend**")
        st.line_chart(df[value_column])
    except Exception as e:
        st.error(f"Error displaying historical data: {str(e)}")

    try:
        st.markdown("**Value Distribution**")
        fig = px.box(df, y=value_column)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error displaying distribution plot: {str(e)}")

    try:
        st.markdown("**Summary Statistics**")
        st.write(df[value_column].describe())
    except Exception as e:
        st.error(f"Error displaying summary statistics: {str(e)}")