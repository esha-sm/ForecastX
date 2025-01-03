import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns

# Load your sales data
file_path = '/Users/esha/downloads/Random_Sales_Dataset.csv'  # Update with the correct path
df = pd.read_csv(file_path)

# Make sure Date is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Fit ARIMA model
model = ARIMA(df['Sales'], order=(5,1,0))  # Adjust p, d, q based on your dataset
model_fit = model.fit()

# Streamlit app
st.title("Sales Forecasting App")

st.write("This app forecasts sales using the ARIMA model.")

# Inform the user about the min and max number of days for prediction
st.write("You can predict sales for a minimum of 1 day and a maximum of 365 days.")

# Display the date range of the historical data in a different font
st.markdown(f"<p style='font-size:20px; color:blue;'>Historical data range: {df['Date'].min().date()} to {df['Date'].max().date()}</p>", unsafe_allow_html=True)

# Input for number of days to forecast
days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

if st.button("Forecast"):
    # Make forecast for the given number of days
    forecast = model_fit.forecast(steps=days)

    # Display the forecast
    st.write(f"Forecast for the next {days} days:")
    st.write(forecast)

    # Plot the forecast
    forecast_dates = pd.date_range(start=df['Date'].iloc[-1], periods=days + 1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    forecast_df.set_index('Date', inplace=True)

    st.line_chart(forecast_df['Forecast'])

    # Combine historical and forecast data for plotting
    combined_df = pd.concat([df.set_index('Date')['Sales'], forecast_df['Forecast']], axis=1)
    combined_df.columns = ['Historical Sales', 'Forecasted Sales']

    # Plot combined historical and forecast data
    st.write("Combined Historical and Forecasted Sales Data")
    st.line_chart(combined_df)

    # Display forecasted mean values with color
    st.write("Forecasted Mean Values")
    styled_forecast_df = forecast_df.style.background_gradient(cmap='coolwarm')
    st.dataframe(styled_forecast_df)

# Display the historical sales data
st.write("Historical Sales Data")
st.line_chart(df.set_index('Date')['Sales'])

# Additional visualizations
st.write("Additional Visualizations")

# Boxplot of sales by product
st.write("Boxplot of Sales by Product")
fig, ax = plt.subplots()
sns.boxplot(x='Product', y='Sales', data=df, palette="Set2", ax=ax)
st.pyplot(fig)

# Barplot of total sales by region
st.write("Total Sales by Region")
total_sales_by_region = df.groupby('Region')['Sales'].sum()
fig, ax = plt.subplots()
sns.barplot(x=total_sales_by_region.index, y=total_sales_by_region.values, palette="Blues_d", ax=ax)
st.pyplot(fig)

