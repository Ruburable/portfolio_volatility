import json
import yfinance as yf
from datetime import datetime, timedelta
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

    # Updated: use initial_date and set end_date = today
    start_date = datetime.strptime(specs['initial_date'], '%Y-%m-%d')
    end_date = datetime.today()
    window = specs['window']

    try:
        prices = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=False)

        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices['Adj Close']
        else:
            prices = prices[['Adj Close']] if 'Adj Close' in prices.columns else prices

        if prices.empty:
            print("Download unsuccessful!")
            return None, None, None, None

    except Exception as e:
        print("Download unsuccessful!", e)
        return None, None, None, None

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

    # === NEW: Calculate performance ===
    # Portfolio daily returns given weights (as of yesterday's close)
    portfolio_returns = returns.dot(weights)
    portfolio_cum_return = (1 + portfolio_returns).cumprod() - 1
    total_return = portfolio_cum_return.iloc[-1]
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan

    performance = pd.DataFrame({
        'date': portfolio_returns.index,
        'daily_return': portfolio_returns.values,
        'cumulative_return': portfolio_cum_return.values
    })

    summary = pd.DataFrame([{
        'portfolio_name': portfolio_name,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d')
    }])

    # Save dataframes to CSV
    calc_in.to_csv('out/out_calculate/calc_in.csv', index=False)
    vol.to_csv('out/out_calculate/vol.csv', index=False)
    corr.to_csv('out/out_calculate/corr.csv')
    performance.to_csv('out/out_calculate/performance_timeseries.csv', index=False)
    summary.to_csv('out/out_calculate/performance_summary.csv', index=False)

    print("Calculation complete!")

    return calc_in, vol, corr, performance, summary


if __name__ == "__main__":
    calc_in, vol, corr, performance, summary = main()