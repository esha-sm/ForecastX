import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('./Random_Sales_Dataset.csv')
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Sales'])

# Fit ARIMA model
model = ARIMA(df['Sales'], order=(5,1,0))
model_fit = model.fit()

st.title("Sales Forecasting App")
st.write("This app forecasts sales using the ARIMA model.")
st.write("You can predict sales for a minimum of 1 day and a maximum of 365 days.")
st.markdown(f"<p style='font-size:20px; color:blue;'>Historical data range: {df['Date'].min().date()} to {df['Date'].max().date()}</p>", unsafe_allow_html=True)

# Input for forecast days
days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

if st.button("Generate Forecast"):
    # Generate forecast for the specified number of days
    forecast = model_fit.forecast(steps=days)
    
    # Create date range for forecast
    last_date = df['Date'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    
    # Create forecast DataFrame with dates
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast
    })
    
    st.write(f"Forecasted Sales for the Next {days} Days")
    st.dataframe(forecast_df.set_index('Date'))
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df['Sales'], label='Historical Sales', alpha=0.7)
    ax.plot(forecast_dates, forecast, label='Forecast', color='red')
    ax.set_title('Sales Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Original visualizations
st.write("Historical Sales Data")
st.line_chart(df.set_index('Date')['Sales'])

st.write("Boxplot of Sales by Product")
fig, ax = plt.subplots()
sns.boxplot(x='Product', y='Sales', data=df, palette="husl", ax=ax)
st.pyplot(fig)

st.write("Total Sales by Region")
total_sales_by_region = df.groupby('Region')['Sales'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(x='Region', y='Sales', data=total_sales_by_region, palette="muted", ax=ax)
ax.set_title("Total Sales by Region", fontsize=16)
ax.set_xlabel("Region", fontsize=14)
ax.set_ylabel("Total Sales", fontsize=14)
ax.grid(False)
sns.despine()
st.pyplot(fig)