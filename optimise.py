import json
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import os


def main():
    os.makedirs('out/out_optimise', exist_ok=True)

    json_file = 'input.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
    portfolio = data['portfolio']
    specs = data['specs']

    tickers = portfolio['tickers']
    current_weights = np.array(portfolio['weights'])
    portfolio_name = portfolio['name']

    start_date = datetime.strptime(specs['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(specs['end_date'], '%Y-%m-%d')

    try:
        prices = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=False)

        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices['Adj Close']
        else:
            prices = prices[['Adj Close']] if 'Adj Close' in prices.columns else prices

        if prices.empty:
            print("Download unsuccessful!")
            return None

    except Exception as e:
        print("Download unsuccessful!")
        return None

    returns = prices.pct_change().dropna()

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    num_portfolios = 100000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(weights * mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_std

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe_ratio

    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]
    optimal_return = results[0, max_sharpe_idx]
    optimal_std = results[1, max_sharpe_idx]
    optimal_sharpe = results[2, max_sharpe_idx]

    current_return = np.sum(current_weights * mean_returns)
    current_std = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))
    current_sharpe = current_return / current_std

    optimal_df = pd.DataFrame({
        'ticker': tickers,
        'current_weight': current_weights,
        'optimal_weight': optimal_weights
    })

    stats_df = pd.DataFrame({
        'portfolio_type': ['Current', 'Optimal'],
        'expected_return': [current_return, optimal_return],
        'volatility': [current_std, optimal_std],
        'sharpe_ratio': [current_sharpe, optimal_sharpe]
    })

    results_df = pd.DataFrame({
        'returns': results[0, :],
        'volatility': results[1, :],
        'sharpe_ratio': results[2, :]
    })

    optimal_df.to_csv('out/out_optimise/optimal_weights.csv', index=False)
    stats_df.to_csv('out/out_optimise/portfolio_stats.csv', index=False)
    results_df.to_csv('out/out_optimise/simulation_results.csv', index=False)

    print("Optimisation complete!")

    return optimal_df, stats_df


if __name__ == "__main__":
    optimal_weights, stats = main()