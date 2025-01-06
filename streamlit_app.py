import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("Data Forecasting App")
st.write("This app forecasts data using the ARIMA model.")
st.write("You can predict values for a minimum of 1 day and a maximum of 365 days.")
st.markdown("<p style='color:red;'>Upload your own file, or if not, no problem! Just use the default dataset to enjoy the app.</p>", unsafe_allow_html=True)

# Option to use default dataset or upload a file
use_default = st.checkbox("Use default dataset")

if use_default:
    try:
        df = pd.read_csv('./Random_Sales_Dataset.csv')
        st.success("Default data loaded successfully!")
    except Exception as e:
        st.error(f"An error occurred while loading the default data: {str(e)}")
else:
    # File uploader for user to upload their own data
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"An error occurred while loading the data: {str(e)}")
    else:
        st.stop()

# Display the first few rows of the dataset
st.write("Here is a preview of your dataset:")
st.dataframe(df.head())

# Allow users to select columns for analysis
columns = df.columns.tolist()
if len(columns) < 2:
    st.error("The dataset must contain at least two columns.")
    st.stop()

date_column = st.selectbox("Select the date column", columns)
value_column = st.selectbox("Select the value column to forecast", columns)

# Handle missing values
missing_value_option = st.selectbox(
    "How would you like to handle missing values?",
    ("Drop rows with missing values", "Fill with mean", "Fill with median")
)

if missing_value_option == "Drop rows with missing values":
    df = df.dropna(subset=[value_column])
elif missing_value_option == "Fill with mean":
    df[value_column] = df[value_column].fillna(df[value_column].mean())
elif missing_value_option == "Fill with median":
    df[value_column] = df[value_column].fillna(df[value_column].median())

try:
    df[date_column] = pd.to_datetime(df[date_column])
except Exception as e:
    st.error(f"An error occurred while processing the date column: {str(e)}")
    st.stop()

try:
    model = ARIMA(df[value_column], order=(5,1,0))
    model_fit = model.fit()
    st.success("ARIMA model fitted successfully!")
except Exception as e:
    st.error(f"An error occurred while fitting the ARIMA model: {str(e)}")
    st.stop()

st.markdown(f"<p style='font-size:20px; color:blue;'>Historical data range: {df[date_column].min().date()} to {df[date_column].max().date()}</p>", unsafe_allow_html=True)

days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

if st.button("Generate Forecast"):
    try:
        forecast = model_fit.forecast(steps=days)
        last_date = df[date_column].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast
        })
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
        st.write(f"Forecasted values for the next {days} days")
        
        def color_code(val):
            if val > forecast_df['Forecast'].mean():
                return 'background-color: lightgreen'
            else:
                return 'background-color: lightcoral'
        
        st.dataframe(forecast_df.style.applymap(color_code, subset=['Forecast']))
        
        fig = px.line(
            title='Forecast',
            template='plotly_white'
        )
        fig.add_scatter(
            x=df[date_column],
            y=df[value_column],
            name='Historical Data',
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
            yaxis_title='Value',
            template='plotly_white'
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while generating the forecast: {str(e)}")

st.markdown("---")
st.header("Additional Visualizations")

try:
    st.markdown("**Historical Data**")
    st.line_chart(df.set_index(date_column)[value_column])
except Exception as e:
    st.error(f"An error occurred while displaying historical data: {str(e)}")

try:
    st.markdown("**Boxplot of Values by Selected Column**")
    selected_column = st.selectbox("Select a column for boxplot", columns)
    fig = px.box(df, x=selected_column, y=value_column, color=selected_column)
    fig.update_layout(
        xaxis_title=selected_column,
        yaxis_title='Value',
        template='plotly_white'
    )
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"An error occurred while displaying the boxplot: {str(e)}")

try:
    st.markdown("**Total Values by Selected Column**")
    selected_column = st.selectbox("Select a column for bar chart", columns, key="bar_chart")
    total_values_by_column = df.groupby(selected_column)[value_column].sum().reset_index()
    fig = px.bar(total_values_by_column, x=selected_column, y=value_column, color=selected_column)
    fig.update_layout(
        xaxis_title=selected_column,
        yaxis_title='Total Value',
        template='plotly_white'
    )
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"An error occurred while displaying total values by column: {str(e)}")