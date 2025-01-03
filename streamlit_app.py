import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

try:
    df = pd.read_csv('./Random_Sales_Dataset.csv')
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"An error occurred while loading the data: {str(e)}")

try:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Sales'])
except Exception as e:
    st.error(f"An error occurred while processing the data: {str(e)}")

try:
    model = ARIMA(df['Sales'], order=(5,1,0))
    model_fit = model.fit()
    st.success("ARIMA model fitted successfully!")
except Exception as e:
    st.error(f"An error occurred while fitting the ARIMA model: {str(e)}")

st.title("Sales Forecasting App")
st.write("This app forecasts sales using the ARIMA model.")
st.write("You can predict sales for a minimum of 1 day and a maximum of 365 days.")
st.markdown(f"<p style='font-size:20px; color:blue;'>Historical data range: {df['Date'].min().date()} to {df['Date'].max().date()}</p>", unsafe_allow_html=True)

days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

if st.button("Generate Forecast"):
    try:
        forecast = model_fit.forecast(steps=days)
        last_date = df['Date'].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast
        })
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
        st.write(f"Forecasted Sales for the Next {days} Days")
        
        def color_code(val):
            if val > forecast_df['Forecast'].mean():
                return 'background-color: lightgreen'
            else:
                return 'background-color: lightcoral'
        
        st.dataframe(forecast_df.style.applymap(color_code, subset=['Forecast']))
        
        fig = px.line(
            title='Sales Forecast',
            template='plotly_white'
        )
        fig.add_scatter(
            x=df['Date'],
            y=df['Sales'],
            name='Historical Sales',
            line=dict(color='blue')
        )
        fig.add_scatter(
            x=forecast_dates,
            y=forecast,
            name='Forecast',
            line=dict(color='red')
        )
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Sales'
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while generating the forecast: {str(e)}")

st.markdown("---")
st.header("Additional Visualizations")

try:
    st.write("Historical Sales Data")
    st.line_chart(df.set_index('Date')['Sales'])
except Exception as e:
    st.error(f"An error occurred while displaying historical sales data: {str(e)}")

try:
    st.write("Boxplot of Sales by Product")
    fig, ax = plt.subplots()
    sns.boxplot(x='Product', y='Sales', data=df, palette="husl", ax=ax)
    ax.set_xlabel("Product")
    ax.set_ylabel("Sales")
    st.pyplot(fig)
except Exception as e:
    st.error(f"An error occurred while displaying the boxplot: {str(e)}")

try:
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
except Exception as e:
    st.error(f"An error occurred while displaying total sales by region: {str(e)}")