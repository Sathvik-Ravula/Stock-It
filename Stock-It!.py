import streamlit as st
import hydralit_components as hc
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import base64
import yfinance as yf
import os
import numpy as np
import psycopg2
import pytz
from datetime import datetime, timedelta
import joblib
from joblib import load
import shap
from alpha_vantage.timeseries import TimeSeries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from pandas_market_calendars import get_calendar
from scipy.stats import norm
import matplotlib.pyplot as plt

# Database connection details
db_host = 'localhost'
db_port = '5432'
db_name = 'test0'
db_user = 'postgres'
db_password = '1234'

# Function to establish a database connection
def create_connection():
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    return conn




st.set_page_config(layout="wide")

menu_data = [
    {'label': "Home"},
    {'label': "EDA"},
    {'label': "Predict Using Our Model"},
    {'label': "Know Our Model"},
    {'label': "Show Model Equation"}
]

over_theme = {'txc_inactive': 'black','menu_background':'skyblue','txc_active':'black','option_active':'white'}

# Create the navbar and get the selected menu item
menu_id = hc.nav_bar(menu_definition=menu_data,hide_streamlit_markers=False,
    sticky_nav=True,
    sticky_mode='pinned',
    override_theme = over_theme)

def fetch_predictions_data(stock_ticker):
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    query = f"SELECT date, actual_close, predicted_close FROM predictions WHERE ticker = '{stock_ticker}';"
    predictions_data = pd.read_sql_query(query, conn)
    conn.close()
    return predictions_data

def load_explainer(stock_ticker):
    # File path for the explainer
    explainer_path = os.path.join("D:/Project-FullStack/TaskAutomation/Models", f"{stock_ticker}_explainer.joblib")
    
    # Check if explainer file exists
    if os.path.exists(explainer_path):
        # Load the explainer
        explainer = load(explainer_path)
        return explainer
    else:
        print(f"Error: Explainer file not found for {stock_ticker}")
        return None


def get_stock_data(stock_ticker):
    try:
        us_timezone = pytz.timezone('US/Eastern')
        today = datetime.now(us_timezone).strftime('%Y-%m-%d')
        # Download data for the current day
        #print(today)
        df = yf.download(stock_ticker, start=today)

        #st.write(df)

        # Check if data is available for the current day
        if df.empty:
            # Display a message indicating a market holiday on the Streamlit app
            st.warning(f"Today is a market holiday. Data not available for {stock_ticker}.")
        else:
            # Data is available for the current day
            # Extract only the 'Open' and 'Date' columns
            df = df[['Open']].copy()
            df.reset_index(inplace=True)
            # Convert 'Date' to 'Y-M-D' format
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

            return df
    except Exception as e:
        st.error(f"Error downloading data for {stock_ticker}: {str(e)}")
        return None



def fetch_stock_data(symbol):
    try:
        # Initialize the Alpha Vantage TimeSeries object
        us_timezone = pytz.timezone('US/Eastern')
        today = datetime.now(us_timezone).strftime('%Y-%m-%d')

        ts = TimeSeries(key='K1V45IRU0NJDL8OA', output_format='pandas', indexing_type='date')
        
        # Fetch intraday data for the stock
        data, meta_data = ts.get_intraday(symbol, interval='1min', outputsize='full')

        # Perform string manipulation on column names
        data.columns = [col[3:] for col in data.columns]

        # Select relevant columns 'open' and 'date' (index)
        stock_data = data[['open']].copy()
        stock_data.index.name = 'Date'
        stock_data.reset_index(inplace=True)

        # Convert 'Date' to datetime format
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])

        # Format 'Date' as 'Y-M-D'
        stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')

        stock_data.rename(columns={'open': 'Open'}, inplace=True)

        return stock_data.head(1)

    except Exception as e:
        print(f"Error fetching data for symbol {symbol} from Alpha Vantage: {str(e)}")
        return None




class LatestRowSelector(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                # Fetch the latest row from the 'stock_data' table
                latest_row = fetch_latest_row_from_database(X)  # Replace with actual function

                return latest_row

def fetch_latest_row_from_database(ticker):
    # Use your database connection and fetch the latest row
    conn = create_connection()

    # Set the timezone to US/Eastern
    us_timezone = pytz.timezone('US/Eastern')
    current_date = datetime.now(us_timezone)

    # If today is Saturday or Sunday, go back to the most recent market day (Friday)
    if current_date.weekday() == 5:  # Saturday
        current_date -= timedelta(days=1)
    elif current_date.weekday() == 6:  # Sunday
        current_date -= timedelta(days=2)
    elif current_date.weekday() == 0:  # Monday
        current_date -= timedelta(days=3)
    else:
        current_date = current_date - timedelta(days=1)

    # Format the dates to string in the required format
    current_date_str = current_date.strftime("%Y-%m-%d")

    # Fetch data for the specified ticker and date
    query = f"SELECT * FROM stock_data WHERE ticker = '{ticker}' AND date = '{current_date_str}';"
    latest_row = pd.read_sql_query(query, conn)

    # If data for the current day is not present, fetch the previous day's data
    if latest_row.empty:
        previous_date = current_date - timedelta(days=1)
        previous_date_str = previous_date.strftime("%Y-%m-%d")
        st.warning("No Sentiment/lag data found for the previous day. Using the most recent available data.")
        query = f"SELECT * FROM stock_data WHERE ticker = '{ticker}' AND date = '{previous_date_str}';"
        latest_row = pd.read_sql_query(query, conn)

        # If data for the previous day is also not present, keep going back until data is found
        target_date_str = '2024-02-10'
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        while latest_row.empty and previous_date >= target_date:

            previous_date -= timedelta(days=1)
            previous_date_str = previous_date.strftime("%Y-%m-%d")
            query = f"SELECT * FROM stock_data WHERE ticker = '{ticker}' AND date = '{previous_date_str}';"
            latest_row = pd.read_sql_query(query, conn)

    conn.close()
    return latest_row


def load_model_and_scalers(selected_ticker):
    # Define the folder path for models
    models_folder_path = r'D:/Project-FullStack/TaskAutomation/Models'

    # Load the Ridge model
    model_path = os.path.join(models_folder_path, f"{selected_ticker}_ridgemodel.joblib")
    ridge_model = load(model_path)

    # Load the MinMaxScaler for features
    scaler_path = os.path.join(models_folder_path, f"{selected_ticker}_scaler.joblib")
    scaler_minmax = load(scaler_path)

    # Load the MinMaxScaler for the target variable 'Close'
    target_scaler_path = os.path.join(models_folder_path, f"{selected_ticker}_target_scaler.joblib")
    scaler_target = load(target_scaler_path)

    print(ridge_model)
    return ridge_model, scaler_minmax, scaler_target

def load_selected_features(selected_ticker, features_folder_path):
    selected_features_file = os.path.join(features_folder_path, f"{selected_ticker}_selected_features.txt")
    with open(selected_features_file, 'r') as file:
        selected_features = [line.strip() for line in file.readlines()]
    return selected_features


def create_connection():
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        return conn

    # Function to execute SQL queries and fetch data
def fetch_data(query, conn):
    try:
        data = pd.read_sql(query, conn)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to get the list of available tickers from the database
def get_available_tickers(conn):
    try:
        query = "SELECT DISTINCT ticker FROM stock_data;"
        tickers = pd.read_sql(query, conn)['ticker'].tolist()
        return tickers
    except Exception as e:
        st.error(f"Error fetching tickers: {str(e)}")
        return []

def is_market_open():
    # Check if it's a weekend
    us_timezone = pytz.timezone('US/Eastern')
    today = datetime.now(us_timezone)
    
    if today.weekday() >= 5:  # Saturday or Sunday
        #st.write("Entered first if")
        return False
    
    # Check if it's a US holiday
    us_calendar = get_calendar("XNYS")
    today = datetime.now(us_timezone).strftime('%Y-%m-%d')
    valid_days_output = us_calendar.valid_days(start_date=today, end_date=today)
    if not any(valid_days_output):
        #st.write("Entered second if")
        return False
    us_timezone = pytz.timezone('US/Eastern')
    today = datetime.now(us_timezone).strftime('%Y-%m-%d')
    # Check if the market is open
    market_ticker = "MSFT"  # S&P 500 can be used as a proxy for the overall market
    market_data = yf.download(market_ticker, start=today)
    if market_data.empty:
        return False
    
    return True

def read_margin_of_error(stock_ticker):
    # File path for margin of error
    file_path = os.path.join("D:/Project-FullStack/TaskAutomation/Models", f"{stock_ticker}_MOE.txt")
    
    # Check if file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            margin_of_error = float(file.readline().strip())  # Read the first line as margin of error
        return margin_of_error
    else:
        print(f"Error: File not found for {stock_ticker}")
        return 0

features_folder_path = r'D:/Project-FullStack/TaskAutomation/Models'

# Define the content for each page
if menu_id == 'Home':
    st.title('Welcome to Stock-It! - our Stock Prediction and Analysis WebApp')
    st.markdown(
        """
        # How Our Model Works

        Our stock price prediction model combines Ridge Regression with sentiment analysis of financial news to provide accurate forecasts. Here's how it works:

        - **Step 1**: Collect historical stock data, including previous close, open, high, low, and adjusted close prices.

        - **Step 2**: Perform sentiment analysis on financial news related to the stock. Extract positive, negative, and neutral sentiments.

        - **Step 3**: Train a Ridge Regression model using historical data and the sentiment scores. Ridge Regression allows us to incorporate various features and prevent overfitting.

        - **Step 4**: Utilize the trained model to make predictions based on user inputs, such as the previous close, open, high, and low prices, along with the extracted sentiment scores.

        - **Step 5**: The model generates a predicted closing price, taking into account both historical trends and current market sentiments.

        ## Key Components

        ### Data Collection

        We collect historical stock data, including previous close, open, high, low, and adjusted close prices, from two primary sources:

        - **Database**: Historical stock data is stored in our database, ensuring easy access to a comprehensive dataset for analysis.

        - **Yahoo Finance API**: For real-time and additional stock data, we use the Yahoo Finance API, automating the process of fetching the latest information.

        ### News Source

        Our news data is sourced from **Google News**, a platform that aggregates news articles and performs clustering to organize related articles in one place. 
        We utilize the GNews Python package ([GNews GitHub Link](https://github.com/ranahaani/GNews)) for web scraping Google News and extracting relevant news articles.

        ### Machine Learning Model

        Our model is built using **Ridge Regression**, a linear regression technique that strikes a balance between accuracy and interpretability. Ridge Regression is well-suited for our stock price prediction task because it not only produces accurate predictions but also provides interpretability in terms of feature importance.

        - **Interpretability**: Ridge Regression assigns weights to features, indicating their importance in predicting the target variable. This allows us to interpret the impact of each feature on the predicted stock price.

        - **Regularization**: Ridge Regression incorporates regularization, preventing overfitting by penalizing large coefficients. This regularization term ensures that the model generalizes well to new data.


        ### Sentiment Analysis

        Financial news sentiment analysis provides additional insights into market sentiment, influencing the predictions.
        Our sentiment analysis is powered by FinBERT-tone, a FinBERT model pre-trained on financial communication text. FinBERT-tone is further fine-tuned 
        on a dataset containing 10,000 manually annotated sentences from analyst reports, achieving superior performance in financial tone analysis.

        More technical details on FinBERT: [FinBERT Technical Details](https://huggingface.co/yiyanghkust/finbert-tone)

        ### Task Automation

        Our web application incorporates task automation to handle various processes, including:

        - **Data Retrieval**: Fetching historical stock data from the database and real-time data from the Yahoo Finance API.

        - **Model Predictions**: Utilizing the Ridge Regression model for accurate stock price predictions.

        - **News Analysis**: Extracting and analyzing news sentiment scores from Google News using the GNews Python package.


        ### Prediction

        Based on the user inputs, historical data, and sentiment scores, our model generates a predicted closing price.

        ### Concept Drift and Data Drift Handling

        To ensure the robustness of our model, we actively address concept drift and data drift:

        - **Daily Data Collection**: We collect new data daily to keep our model up-to-date with the latest market trends.

        - **Daily Model Retraining**: Our model is retrained daily using the most recent data, preventing degradation in performance due to concept drift.

        - **Feature Selection Stability**: Selected features for each stock are saved and monitored for stability. Any significant change prompts a reassessment of the model.


        ### Benefits

        - **Accurate Predictions**: Combining Ridge Regression with sentiment analysis enhances the accuracy of stock price predictions.

        - **Comprehensive Insights**: Users gain a comprehensive understanding of market trends by considering both historical data and current sentiments.

        ### Get Started

        Use the navigation menu above to explore the different sections of our app. Whether you're interested in EDA, stock price predictions, or understanding the model, we've got you covered.

        Happy exploring!
        """
    )



elif menu_id == 'EDA':
    st.title('Exploratory Data Analysis (EDA) Page')
    # Function to establish a database connection
    #st.title('Exploratory Data Analysis (EDA) Page')

    # Establish a connection to the database
    conn = create_connection()

    if conn:
        # Get the list of available tickers
            available_tickers = get_available_tickers(conn)

            # Dropdown to select a stock
            selected_ticker = st.selectbox('Select a Stock Ticker', available_tickers)

            # Define your SQL query to fetch data for the selected ticker
            eda_query = f"SELECT * FROM stock_data WHERE ticker = '{selected_ticker}';"

            # Fetch data from the database when the user clicks the button
            if st.button('Analyze'):
                data = fetch_data(eda_query, conn)
                data = data.sort_values(by='date')

                if data is not None:
                    # Display the raw data
                    st.subheader('Raw Data')
                    st.write(data)

                    # Summary Statistics
                    st.subheader('Summary Statistics')
                    st.write(data.describe())

                    # Line Chart of Closing Price Over Time
                    st.subheader('Closing Price Over Time')
                    fig = px.line(data, x='date', y='close', title=f'Closing Price Over Time - {selected_ticker}')
                    st.plotly_chart(fig)

                    # Correlation Heatmap
                    st.subheader('Correlation Heatmap')
                    correlation_matrix = data.corr()
                    fig = go.Figure(data=go.Heatmap(
                        z=correlation_matrix,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        colorscale='reds'))
                    fig.update_layout(title='Correlation Heatmap')
                    st.plotly_chart(fig)

                    # Candlestick Chart
                    st.subheader('Candlestick Chart (Recent 30 Days)')
                    last_30_days_data = data.tail(30)
                    fig = go.Figure(data=[go.Candlestick(x=last_30_days_data['date'],
                                open=last_30_days_data['open'],
                                high=last_30_days_data['high'],
                                low=last_30_days_data['low'],
                                close=last_30_days_data['close'])])
                    st.plotly_chart(fig)

                    # Daily Returns Histogram
                    st.subheader('Daily Returns Histogram')
                    data['Daily Returns'] = data['close'].pct_change()
                    fig = px.histogram(data, x='Daily Returns', nbins=50, title='Daily Returns Histogram')
                    st.plotly_chart(fig)

                    st.subheader('Sentiment Scores Radar Chart')
                    if 'date' in data.columns and 'positivescore' in data.columns and 'negativescore' in data.columns and 'neutralscore' in data.columns:
                        data['date'] = pd.to_datetime(data['date'])
                        radar_data = data[['date', 'positivescore', 'negativescore', 'neutralscore']]
                        radar_data['date_str'] = radar_data['date'].dt.strftime('%Y-%m-%d')

                        # Create a radar chart using go.Scatterpolar
                        fig = go.Figure()

                        for index, row in radar_data.iterrows():
                            fig.add_trace(go.Scatterpolar(
                                r=row[['positivescore', 'negativescore', 'neutralscore']],
                                theta=['Positive Score', 'Negative Score', 'Neutral Score'],
                                fill='toself',
                                name=row['date_str']
                            ))

                        # Update legend entries with dates
                        fig.update_layout(legend_title_text='Date')
                        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
                        fig.update_layout(title=f'Sentiment Scores Radar Chart - {selected_ticker}')
                        fig.update_layout(legend=dict(traceorder='reversed'))
                        
                        st.plotly_chart(fig)
                    else:
                        st.warning("Sentiment scores are not available for the selected stock.")

            # Close the database connection
            conn.close()
    else:
        st.error("Failed to establish a connection to the database. Please check your database credentials.")


elif menu_id == 'Predict Using Our Model':
    # Define the folder path for models
    models_folder_path = r'D:/Project-FullStack/TaskAutomation/Models'

    # Get a list of available stock tickers from the models folder
    available_tickers = [file.split('_ridgemodel')[0] for file in os.listdir(models_folder_path) if file.endswith('_ridgemodel.joblib')]


    # Add a dropdown for selecting stock tickers
    selected_ticker = st.selectbox('Select Stock Ticker', available_tickers)

    # Check if a stock ticker is selected
    if selected_ticker:
        st.title(f'Stock Price Prediction - {selected_ticker}')

        # Fetch stock data using fetch_stock_data function
        stock_data = get_stock_data(selected_ticker)

        print(stock_data)

        # Fetch the latest row from the 'stock_data' table
        latest_row = fetch_latest_row_from_database(selected_ticker)  # Replace with actual function
        #print(latest_row)

        # Combine outputs from fetch_stock_data and fetch_latest_row_from_database
        

        if is_market_open():
            if st.button('Predict'):

                combined_data = pd.DataFrame({
                    'Date': stock_data['Date'].tolist(),
                    'Open': stock_data['Open'].tolist(),
                    'lPositiveScore': [latest_row['positivescore'].values[0]],
                    'lag_vol': [latest_row['volume'].values[0]],
                    'lNegativeScore': [latest_row['negativescore'].values[0]],
                    'lNeutralScore': [latest_row['neutralscore'].values[0]],
                    'lag_high': [latest_row['high'].values[0]],
                    'lag_close': [latest_row['close'].values[0]],
                    'lag_open' : [latest_row['open'].values[0]],
                    'lag_low' : [latest_row['low'].values[0]],
                })

                features_folder_path = r'D:/Project-FullStack/TaskAutomation/Models'

                selected_features = load_selected_features(selected_ticker, features_folder_path)

                st.write("Selected Features Using SelectKBest :")
                st.write(selected_features)

                # Filter the combined_data DataFrame based on selected features
                selected_data = combined_data[selected_features]
                combined_data1 = combined_data[selected_features]
                combined_data1['Date'] = combined_data['Date']
                st.write(combined_data1)

                # Load the model and scalers
                ridge_model, scaler_minmax, scaler_target = load_model_and_scalers(selected_ticker)

                if ridge_model and scaler_minmax and scaler_target:

                    scaled_data = pd.DataFrame(scaler_minmax.transform(selected_data), columns=selected_data.columns)
                
                    # Print the scaled_data
                    st.write("Scaled Data: ")

                    st.write(scaled_data)

                    # Make predictions using the loaded model
                    predictions = ridge_model.predict(scaled_data)

                    # Inverse transform the predictions using the target scaler
                    original_predictions = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()

                    margin_of_error = float(read_margin_of_error(selected_ticker))

                    z_value = 1.96  # Z-value for a 95% confidence level (two-tailed test)
                    margin_of_error *= z_value

                    lower_bounds = predictions - margin_of_error
                    upper_bounds = predictions + margin_of_error

                    # Inverse scale the bounds
                    lower_bounds_scaled = scaler_target.inverse_transform(lower_bounds.reshape(-1, 1)).flatten()
                    upper_bounds_scaled = scaler_target.inverse_transform(upper_bounds.reshape(-1, 1)).flatten()

                    # Print confidence intervals
                    # Display the predictions or do further processing as needed
                    st.markdown(
                        f"""
                        <div style="background-color: #3498db; padding: 10px; border-radius: 10px;">
                            <p style="font-size: 24px; color: #ffffff; text-align: center;">Predicted Closing Price</p>
                            <p style="font-size: 36px; color: #ffffff; text-align: center;">${original_predictions[0]:.2f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Display confidence interval below predicted price
                    st.markdown(
                        f"""
                        <div style="margin-top: 20px; background-color: #85C1E9; padding: 10px; border-radius: 10px;">
                            <p style="font-size: 24px; color: #000000; text-align: center;">95% Prediction Interval</p>
                            <p style="font-size: 30px; color: #000000; text-align: center;">[{lower_bounds_scaled[0]:.2f}, {upper_bounds_scaled[0]:.2f}]</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    previous_predictions_data = fetch_predictions_data(selected_ticker)

                    previous_predictions_data = pd.DataFrame(previous_predictions_data)
                    # Combine previous predictions and the current prediction
                    all_predictions_data = pd.concat([previous_predictions_data, pd.DataFrame({'date': [datetime.now().strftime('%Y-%m-%d')],
                                                                                              'predicted_close': [original_predictions[0]]})])

                    #st.write(all_predictions_data)

                    all_predictions_data['date'] = pd.to_datetime(all_predictions_data['date'])

                    all_predictions_data = all_predictions_data.sort_values(by='date')

                    # Create an interactive line chart using Plotly
                    fig = px.line(all_predictions_data, x='date', y='predicted_close', title='Predicted Close Price History')

                    # Highlight the latest data point
                    fig.add_trace(go.Scatter(x=[datetime.now().strftime('%Y-%m-%d')],
                                             y=[original_predictions[0]],
                                             mode='markers',
                                             marker=dict(color='red', size=10),
                                             name="Our Model's Prediction"))

                    # Customize chart aesthetics
                    fig.update_traces(line=dict(width=2))  # Adjust line width
                    fig.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size=20)))

                    # Show the Plotly chart in Streamlit
                    st.plotly_chart(fig)

                else:
                    st.warning(f"Failed to load model or scalers for {selected_ticker}. Please check the model and scaler files.")

        else:
            st.warning("Market is closed today. Prediction is disabled.")




    

elif menu_id == 'Know Our Model':
    st.title('Know Our Model Through Performance On Historical Data')

    # Get a list of available stock tickers from the predictions table
    available_tickers = ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "META", "INTC", "AMD", "HSBC", "TSLA", "V", "JPM", "WMT", "MA", "XOM", "HD", "PG", "COST", "PEP"]
    
    # Add a dropdown for selecting stock tickers
    selected_ticker = st.selectbox('Select Stock Ticker', available_tickers)

    # Check if a stock ticker is selected
    if selected_ticker:
        st.title(f'Performance of Model - {selected_ticker}')

        # Fetch predictions data for the selected stock ticker
        predictions_data = fetch_predictions_data(selected_ticker)

        # Create separate traces for actual and predicted close lines
        trace_actual = go.Scatter(x=predictions_data['date'], y=predictions_data['actual_close'],
                                 mode='lines', name='Actual Close', line=dict(width=2, color='blue'))

        trace_predicted = go.Scatter(x=predictions_data['date'], y=predictions_data['predicted_close'],
                                    mode='lines', name='Predicted Close', line=dict(width=2, color='red'))

        # Create an interactive line chart using Plotly with the separate traces
        fig = go.Figure([trace_actual, trace_predicted])
        fig.update_layout(title=f'Actual vs Predicted Close Price for {selected_ticker}',
                          legend=dict(title_font_family="Times New Roman", font=dict(size=20)))

        # Show the Plotly chart in Streamlit
        st.plotly_chart(fig)
            

elif menu_id == 'Show Model Equation':

    available_tickers = ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "META", "INTC", "AMD", "HSBC", "TSLA", "V", "JPM", "WMT", "MA", "XOM", "HD", "PG", "COST", "PEP"]
    
    # Add a dropdown for selecting stock tickers
    selected_ticker = st.selectbox('Select Stock Ticker', available_tickers)

    ridge_model, _, _ = load_model_and_scalers(selected_ticker)
    selected_features = load_selected_features(selected_ticker, features_folder_path)
    # Print the beta values
    intercept = ridge_model.intercept_
    coefficients = ridge_model.coef_.flatten()

    # Create a LaTeX-formatted model equation
    selected_features = ['Intercept'] + selected_features
    model_equation = f"{intercept:.4f} + " + " + ".join([f"({coefficient:.4f}) \\times {feature}" for feature, coefficient in zip(selected_features[1:], coefficients)])

    # Display explanatory text
    st.markdown(
        """
        ## Model Equation

        The Ridge Regression model used in our prediction combines historical stock data and sentiment scores to generate predictions.
        The model equation is as follows:

        Predicted Closing Price = Intercept + Coefficient1 * Scaled Open + ... + Coefficient7 * Scaled Lag Close

        Where:
        - Intercept, Coefficient1 to Coefficient7 are the coefficients learned by the Ridge Regression model.
        - Scaled Open, ..., Scaled Lag Close are the scaled features used in the prediction.

        This equation captures the relationship between various features and the predicted closing price, providing transparency into the model's decision-making process.

        ### Interpretation

        - Coefficient1 indicates the impact of the scaled Open price on the predicted closing price.
        - Similarly, each coefficient (Coefficient2 to Coefficient7) corresponds to the impact of the respective feature on the prediction.

        Understanding the coefficients helps interpret the significance of each feature in predicting stock prices.
        """
    )


    # Display the LaTeX equation
    st.markdown(f"$$Predicted Closing Price = {model_equation}$$", unsafe_allow_html=True)
    st.info("Note that the Model works on Scaled Data")


