from datetime import datetime
import argparse
import os
import joblib

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set(style="darkgrid")


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance using yfinance."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker} between {start} and {end}")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index.name = 'Date'
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for prediction.
    Features included:
    - Previous day's Close (lag1)
    - Lag returns for 1, 2, 3 days
    - Moving averages 5, 10, 21 days
    - Volume and volume change
    - High-Low range
    Target:
    - Next day's Close (Close shifted -1)
    """
    df = df.copy()
    df['Close_lag1'] = df['Close'].shift(1)
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_2'] = df['Close'].pct_change(2)
    df['Return_3'] = df['Close'].pct_change(3)

    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()

    df['Vol_change_1'] = df['Volume'].pct_change(1)
    df['HL_range'] = (df['High'] - df['Low']) / df['Open']

    # Target: next day's close
    df['Target'] = df['Close'].shift(-1)

    # Drop rows with NaNs created by shifting/rolling
    df = df.dropna().copy()
    return df


def prepare_data_for_model(df: pd.DataFrame, feature_cols: list):
    """Split into X and y. Scale numeric features.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    X = df[feature_cols].values
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )  # preserve time order -> no shuffling for time-series simple split

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test):
    """Train models and return dictionary of results including predictions and metrics."""
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {
            'model': model,
            'preds': preds,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    return results


def plot_predictions(df: pd.DataFrame, X_test_index, y_test, results: dict, ticker: str, outdir: str):
    """Plot actual vs predicted closing prices for each model."""
    plt.figure(figsize=(12, 6))
    plt.plot(X_test_index, y_test, label='Actual', linewidth=2)

    for name, res in results.items():
        plt.plot(X_test_index, res['preds'], label=f'Predicted ({name})', alpha=0.9)

    plt.title(f'{ticker} - Actual vs Predicted Close')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(outdir, f'{ticker}_predictions.png')
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f'Plot saved to {outpath}')


def print_results(results: dict):
    """Print evaluation metrics in a neat table-like format."""
    rows = []
    for name, r in results.items():
        rows.append({
            'Model': name,
            'RMSE': round(r['rmse'], 4),
            'MAE': round(r['mae'], 4),
            'R2': round(r['r2'], 4)
        })
    res_df = pd.DataFrame(rows).set_index('Model')
    print('\nEvaluation Results:\n')
    print(res_df)


def save_models_and_artifacts(results: dict, scaler, feature_cols: list, outdir: str, ticker: str):
    os.makedirs(outdir, exist_ok=True)
    # Save scaler
    joblib.dump(scaler, os.path.join(outdir, f'{ticker}_scaler.pkl'))
    # Save feature names
    joblib.dump(feature_cols, os.path.join(outdir, f'{ticker}_feature_cols.pkl'))

    for name, r in results.items():
        joblib.dump(r['model'], os.path.join(outdir, f'{ticker}_{name}.pkl'))

    print(f'Models and artifacts saved to {outdir}')


def parse_args():
    parser = argparse.ArgumentParser(description='Stock Price Prediction Example')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date YYYY-MM-DD')
    parser.add_argument('--outdir', type=str, default='artifacts', help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()

    print(f'Downloading data for {ticker} from {args.start} to {args.end}...')
    df = download_data(ticker, args.start, args.end)

    print('Creating features...')
    df_feat = create_features(df)

    # Choose features
    feature_cols = [
        'Close_lag1', 'Return_1', 'Return_2', 'Return_3',
        'MA_5', 'MA_10', 'MA_21', 'Vol_change_1', 'HL_range'
    ]

    # Keep index to map test ranges to dates
    split_index = int(len(df_feat) * 0.8)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data_for_model(df_feat, feature_cols)

    # We'll also need the date index for the test set when plotting
    dates = df_feat.index
    test_dates = dates[split_index:]

    # Define models
    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(max_depth=6, random_state=42)
    }

    print('Training models...')
    results = train_and_evaluate(models, X_train_scaled, X_test_scaled, y_train, y_test)

    print_results(results)

    print('Saving models and artifacts...')
    save_models_and_artifacts(results, scaler, feature_cols, args.outdir, ticker)

    print('Plotting predictions...')
    plot_predictions(df_feat, test_dates, y_test, results, ticker, args.outdir)

    print('\nDone!')


if __name__ == '__main__':
    main()
