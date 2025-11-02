import json
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import os


def main():
    os.makedirs('out/out_calculate', exist_ok=True)

    json_file = 'input.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
    portfolio = data['portfolio']
    specs = data['specs']

    tickers = portfolio['tickers']
    weights = np.array(portfolio['weights'])
    portfolio_name = portfolio['name']

    start_date = datetime.strptime(specs['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(specs['end_date'], '%Y-%m-%d')
    window = specs['window']

    try:
        prices = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=False)

        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices['Adj Close']
        else:
            prices = prices[['Adj Close']] if 'Adj Close' in prices.columns else prices

        if prices.empty:
            print("Download unsuccessful!")
            return None, None, None

    except Exception as e:
        print("Download unsuccessful!")
        return None, None, None

    returns = prices.pct_change().dropna()

    # Store calculation details
    calc_details = {
        'portfolio_name': portfolio_name,
        'tickers': tickers,
        'weights': weights,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'window': window,
        'trading_days': len(returns)
    }
    calc_in = pd.DataFrame([calc_details])

    # Calculate rolling portfolio volatility
    rolling_volatility = []
    dates = []

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i - window:i]
        cov_matrix = window_returns.cov()
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        annualized_vol = portfolio_vol * np.sqrt(252)

        rolling_volatility.append(annualized_vol)
        dates.append(returns.index[i])

    volatility = pd.Series(rolling_volatility, index=dates)
    vol = pd.DataFrame({'date': volatility.index, 'volatility': volatility.values})
    vol.reset_index(drop=True, inplace=True)

    # Create correlation matrix
    correlation_matrix = returns.corr()
    corr = correlation_matrix.copy()

    # Save dataframes to CSV
    calc_in.to_csv('out/out_calculate/calc_in.csv', index=False)
    vol.to_csv('out/out_calculate/vol.csv', index=False)
    corr.to_csv('out/out_calculate/corr.csv')

    print("Calculation complete!")

    return calc_in, vol, corr


if __name__ == "__main__":
    calc_in, vol, corr = main()