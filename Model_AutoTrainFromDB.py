import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from joblib import dump, load
from datetime import datetime
import psycopg2
import shap

# Define your PostgreSQL database connection parameters
db_host = 'localhost'
db_port = '5432'
db_name = 'test0'
db_user = 'postgres'
db_password = '1234'

def create_predictions_table():
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    
    # Drop the table if it already exists
    cur.execute("DROP TABLE IF EXISTS predictions;")
    
    # Create the predictions table
    cur.execute("""
        CREATE TABLE predictions (
            ticker VARCHAR(10),
            date DATE,
            actual_close FLOAT,
            predicted_close FLOAT
        );
    """)

    conn.commit()
    conn.close()

def insert_predictions_into_table(ticker, date, actual_close, predicted_close):
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    cursor = conn.cursor()

    # Insert predictions into the table
    insert_query = f'''
        INSERT INTO predictions (ticker, date, actual_close, predicted_close)
        VALUES ('{ticker}', '{date}', {actual_close}, {predicted_close});
    '''
    cursor.execute(insert_query)

    # Commit changes and close connection
    conn.commit()
    conn.close()

def save_margin_of_error(stock_ticker, margin_of_error):
    # Save the margin of error for the current stock to a text file
    with open(f'D:/Project-FullStack/TaskAutomation/Models/{stock_ticker}_MOE.txt', 'w') as file:
        file.write(f'{margin_of_error}')

def display_metrics(y_test, y_pred, model_name, stock_ticker):
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    print(f"{model_name} Metrics:")
    print(f"MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R2 Score: {r2}\n")

    # Save margin of error
    margin_of_error = np.abs(y_pred - y_test).mean()  # Mean Absolute Error is the margin of error in this case
    save_margin_of_error(stock_ticker, margin_of_error)

    return mae, mse, rmse, r2

def fetch_latest_data_from_database(stock_ticker, num_rows=252):
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    query = f"SELECT * FROM stock_data WHERE Ticker = '{stock_ticker}' ORDER BY Date DESC LIMIT {num_rows};"
    latest_data = pd.read_sql_query(query, conn)
    conn.close()
    return latest_data

def main():
    # Directory paths
    model_dir = "D:/Project-FullStack/TaskAutomation/Models"

    # Create a txt file to store metrics
    metrics_file_path = os.path.join(model_dir, "metrics.txt")
    metrics_file = open(metrics_file_path, "w")  # Append mode to add to the existing file

    # Iterate over stock tickers
    stock_tickers = ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "META", "INTC", "AMD", "HSBC", "TSLA", "V", "JPM", "WMT", "MA", "XOM", "HD", "PG", "COST", "PEP"]

    create_predictions_table()

    
    for stock_ticker in stock_tickers:
        # Fetch latest data from the database
        latest_data = fetch_latest_data_from_database(stock_ticker)

        # Feature engineering
        data = latest_data.drop("date", axis=1)

        # Create lag features
        latest_data['Open'] = latest_data['open']
        latest_data['Date'] = latest_data['date']
        latest_data['Close'] = latest_data['close']
        latest_data['lag_close'] = latest_data['close'].shift(1)
        latest_data['lag_high'] = latest_data['high'].shift(1)
        latest_data['lag_low'] = latest_data['low'].shift(1)
        latest_data['lag_vol'] = latest_data['volume'].shift(1)
        latest_data['lag_open'] = latest_data['open'].shift(1)
        latest_data['lPositiveScore'] = latest_data['positivescore'].shift(1)
        latest_data['lNegativeScore'] = latest_data['negativescore'].shift(1)
        latest_data['lNeutralScore'] = latest_data['neutralscore'].shift(1)

        # Dropping unnecessary columns
        latest_data = latest_data[['Date', 'Open', 'lPositiveScore', 'lag_vol', 'lNegativeScore', 'lNeutralScore','lag_high','lag_close', 'Close']]

        # Drop the first row (NaN due to lag)
        latest_data.dropna(inplace=True)

        # Feature selection using best subset selection
        X = latest_data.drop(["Date", "Close"], axis=1)  # Features excluding Date and the target variable (lag_close)
        y = latest_data['Close']  # Target variable

        # Find the best features using best subset selection
        selector = SelectKBest(score_func=f_regression, k=7)
        selector.fit(X, y)
        best_feature_indices = selector.get_support(indices=True)

        # Select the best features
        selected_features = X.columns[best_feature_indices]

        # Save the selected features for each stock name
        features_file_path = os.path.join(model_dir, f"{stock_ticker}_selected_features.txt")
        with open(features_file_path, "w") as features_file:
            features_file.write("\n".join(selected_features))

        # Apply Min-Max Scaling to selected features
        scaler_minmax = MinMaxScaler()
        latest_data[selected_features] = scaler_minmax.fit_transform(latest_data[selected_features])

        # Apply Min-Max Scaling to the target variable 'Close' with a different scaler instance
        scaler_target = MinMaxScaler()
        original_close_values = latest_data['Close'].values
        latest_data['Close'] = scaler_target.fit_transform(latest_data[['Close']])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(latest_data[selected_features], latest_data['Close'], test_size=0.05, random_state=2)

        #Linear Splitting
        #latest_data = latest_data.sort_values(by='Date')

        # Calculate the index to split the data
        #split_index = int(0.9 * len(latest_data))

        # Split the data
        #training_data = latest_data.iloc[:split_index]
        #testing_data = latest_data.iloc[split_index:]

        # Split into features and target variable
        #X_train = training_data[selected_features]
        #y_train = training_data['Close']

        #X_test = testing_data[selected_features]
        #y_test = testing_data['Close']

        # Ridge Regression model
        ridge_model = Ridge(alpha=1.5)
        ridge_model.fit(X_train, y_train)

        explainer = shap.Explainer(ridge_model, X_train)
        shap_values = explainer.shap_values(X_train)
        
        # Make predictions
        y_pred = ridge_model.predict(X_test)

        # Display metrics for the current model
        mae, mse, rmse, r2 = display_metrics(y_test, y_pred, f"{stock_ticker} Ridge Model", stock_ticker)

        # Save metrics to the txt file
        metrics_file.write(f"\n Date: {datetime.now().strftime('%Y-%m-%d')}\n")
        metrics_file.write(f"{stock_ticker} - MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R2 Score: {r2}\n")

        # Save the Ridge model and MinMaxScaler instance, overwriting if they already exist
        model_path = os.path.join(model_dir, f"{stock_ticker}_ridgemodel.joblib")
        scaler_path = os.path.join(model_dir, f"{stock_ticker}_scaler.joblib")
        scaler_target_path = os.path.join(model_dir, f"{stock_ticker}_target_scaler.joblib")
        explainer_path = os.path.join(model_dir,f"{stock_ticker}_explainer.joblib")

        dump(ridge_model, model_path, compress=True)
        dump(scaler_minmax, scaler_path, compress=True)
        dump(scaler_target, scaler_target_path, compress=True)
        dump(explainer, explainer_path)

        X_all = latest_data[selected_features]
        predictions_all = ridge_model.predict(X_all)

        # Inverse transform the predictions using the target scaler
        original_predictions_all = scaler_target.inverse_transform(predictions_all.reshape(-1, 1)).flatten()
        latest_data['Close'] = original_close_values

        # Insert predictions into the predictions table for the entire dataset
        for date, actual_close, predicted_close, features_row in zip(latest_data['Date'], latest_data['Close'], original_predictions_all, X_all.iterrows()):
            insert_predictions_into_table(stock_ticker, date, actual_close, predicted_close)

    # Close the metrics file
    metrics_file.close()

if __name__ == "__main__":
    main()
