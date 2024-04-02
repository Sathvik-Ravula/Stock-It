import os
import pandas as pd
from datetime import datetime, timedelta
from gnews import GNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import yfinance as yf
import time
import psycopg2

# Define your PostgreSQL database connection parameters
db_host = 'localhost'
db_port = '5432'
db_name = 'test0'
db_user = 'postgres'
db_password = '1234'

def create_database():
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    cursor = conn.cursor()

    # Create a table to store stock data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            Date DATE PRIMARY KEY,
            Ticker VARCHAR(10),
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume INTEGER,
            PositiveScore REAL,
            NegativeScore REAL,
            NeutralScore REAL
        );
    ''')

    conn.commit()
    conn.close()

from sqlalchemy import create_engine

def save_combined_data_to_database(stock_ticker_combined, stock_ticker):
    # Establish a connection to the database
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'test0'
    db_user = 'postgres'
    db_password = '1234'
    
    # Construct the database URL
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Create a SQLAlchemy engine
    engine = create_engine(db_url)
    stock_ticker_combined.columns = stock_ticker_combined.columns.str.lower()
    stock_ticker_combined.index.names = ['date']
    stock_ticker_combined = stock_ticker_combined.rename(columns={'adj close': 'adj_close'})

    # Save the combined DataFrame to the database
    stock_ticker_combined.to_sql('stock_data', engine, index=True, if_exists='append', method='multi')

    # Close the SQLAlchemy engine
    engine.dispose()

def fetch_stock_data_from_database(stock_ticker):
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    query = f"SELECT * FROM stock_data WHERE Ticker = '{stock_ticker}';"
    df = pd.read_sql_query(query, conn, index_col='Date', parse_dates=['Date'])
    conn.close()
    return df

def get_stock_data(stock_ticker, start_date, end_date):
    df = yf.download(stock_ticker, start=start_date, end=end_date)
    return df

def get_stock_news(stock_ticker, stock_name, start_date, end_date):
    news = []
    current_date = datetime(*start_date)

    google_news = GNews(language='en', country='US',start_date=current_date, end_date=end_date)
    news = google_news.get_news(f'{stock_name} ,{stock_ticker}')
    print(f"{len(news)} articles generated between {current_date} and {end_date} for {stock_ticker}")

    return news


# Function to preprocess news data
def preprocess_news_data(news_data, stock_ticker):
    df = pd.DataFrame(news_data)
    input_format = "%a, %d %b %Y %H:%M:%S GMT"
    output_format = "%Y-%m-%d"
    df['published date'] = pd.to_datetime(df['published date'], format=input_format)
    df['published date'] = df['published date'].dt.strftime(output_format)

    df = df.sort_values(by='published date', ascending=False)
    df.set_index('published date', inplace=True)
    df['title'] = df['title'].str.lower()
    df = df.drop(["url", "publisher"], axis=1)
    df.rename(columns={'published date': 'Date'}, inplace=True)

    return df

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Function to analyze sentiment for news descriptions
def analyze_sentiment(description, stock_ticker):
    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    positive_score = probabilities[0, 0].item()
    negative_score = probabilities[0, 1].item()
    neutral_score = probabilities[0, 2].item()

    return {
        f'PositiveScore': positive_score,
        f'NegativeScore': negative_score,
        f'NeutralScore': neutral_score
    }

def main():
    create_database()
    
    stock_tickers_and_names = [
        ("AAPL", "Apple"),
        ("MSFT", "Microsoft"),
        ("GOOG", "Alphabet"),
        ("GOOGL", "Google"),
        ("AMZN", "Amazon"),
        ("NVDA", "NVIDIA"),
        ("META", "Meta Platforms"),
        ("INTC", "Intel Corporation"),
        ("AMD", "Advanced Micro Devices"),
        ("HSBC", "HSBC Holdings"),
        ("TSLA", "Tesla"),
        ("V", "Visa"),
        ("JPM", "JP Morgan Chase"),
        ("WMT", "Walmart"),
        ("MA", "Mastercard"),
        ("JNJ", "Johnson & Johnson"),
        ("XOM", "Exxon Mobil"),
        ("HD", "Home Depot"),
        ("PG", "Procter & Gamble"),
        ("COST", "Costco"),
        ('PEP', "Pepsi Co")
    ]

    folder_path = "D:\Project-FullStack\TaskAutomation"
    os.makedirs(folder_path, exist_ok=True)

    for stock_ticker, stock_name in stock_tickers_and_names:
        # Fetch the latest date from the database
        latest_date = fetch_latest_date_from_database(stock_ticker)
        
        if latest_date is not None:
            # Set the start date and end date
            start_date = latest_date + timedelta(days=1)
            end_date = (datetime.now() + timedelta(days=1))
            # Get stock price data
            stock_data = get_stock_data(stock_ticker, start_date, end_date)
            print(stock_data)
            stock_data.index = pd.to_datetime(stock_data.index)

            if not stock_data.empty:
                # Get news data
                start_date = (start_date.year, start_date.month, start_date.day)
                print(start_date)
                end_date = (end_date.year, end_date.month, end_date.day)
                print(end_date)
                news_data = get_stock_news(stock_ticker, stock_name, start_date, end_date)

                if len(news_data) != 0:
                    # Preprocess news data
                    df = preprocess_news_data(news_data, stock_ticker)
                    print(df)
                    # Analyze sentiment
                    sentiment_scores = df['title'].apply(lambda x: analyze_sentiment(x, stock_ticker))
                    sentiment_df = pd.DataFrame(list(sentiment_scores))
                    sentiment_df['Date'] = df.index
                    sentiment_means = sentiment_df.groupby(sentiment_df['Date']).mean()
                    sentiment_means.index = pd.to_datetime(sentiment_means.index)

                    # Merge stock price data and sentiment data
                    stock_ticker_combined = pd.merge(stock_data, sentiment_means, left_index=True, right_index=True, how='inner')
                    stock_ticker_combined['ticker'] = stock_ticker

                    print("***************************\n\n\n")
                    print(stock_ticker_combined)
                    print("\n\n\n**************************")

                    # Save the combined DataFrame to the database
                    save_combined_data_to_database(stock_ticker_combined, stock_ticker)

                    remove_null_ticker_rows(stock_ticker)

                    # Remove duplicate rows (same date and ticker) directly from the database
                    remove_duplicate_rows(stock_ticker)

                else:
                    print("Data not generated!")
                    pass
            else:
                print("No Stock Data Found For Given Day")
                

def fetch_latest_date_from_database(stock_ticker):
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    query = f"SELECT MAX(Date) FROM stock_data WHERE Ticker = '{stock_ticker}';"
    latest_date = pd.read_sql_query(query, conn).iloc[0, 0]
    conn.close()
    return latest_date

def remove_null_ticker_rows(stock_ticker):
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    cursor = conn.cursor()

    query = f"DELETE FROM stock_data WHERE Ticker IS NULL AND Ticker = '{stock_ticker}';"
    cursor.execute(query)

    conn.commit()
    conn.close()

def remove_duplicate_rows(stock_ticker):
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    cursor = conn.cursor()

    query = f"DELETE FROM stock_data a USING stock_data b WHERE a.Date = b.Date AND a.Ticker = b.Ticker AND a.ctid < b.ctid AND a.Ticker = '{stock_ticker}';"
    cursor.execute(query)

    conn.commit()
    conn.close()




if __name__ == "__main__":
    main()
