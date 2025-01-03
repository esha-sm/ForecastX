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

# Ensure there are no missing values in the Sales column
df = df.dropna(subset=['Sales'])

# Fit the ARIMA model
model = ARIMA(df['Sales'], order=(5,1,0))
model_fit = model.fit()

# Forecast the next 30 days
forecast = model_fit.forecast(steps=30)
forecast_df = pd.DataFrame(forecast, columns=['Forecast'])

# Display the forecasted values
st.write("Forecasted Sales for the Next 30 Days")
st.dataframe(forecast_df)

# Plot the forecasted values
st.line_chart(forecast_df)

# Additional visualizations
st.write("Historical Sales Data")
st.line_chart(df.set_index('Date')['Sales'])

st.write("Boxplot of Sales by Product")
fig, ax = plt.subplots()
sns.boxplot(x='Product', y='Sales', data=df, palette="husl2", ax=ax)
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