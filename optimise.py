import json
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    num_portfolios = 10000
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

    current_return = np.sum(current_weights * mean_returns)
    current_std = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))

    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.3, s=10)
    plt.colorbar(label='Sharpe Ratio')

    plt.scatter(optimal_std, optimal_return, marker='*', color='red', s=500,
                edgecolors='black', linewidth=2, label='Optimal Portfolio', zorder=5)

    plt.scatter(current_std, current_return, marker='o', color='blue', s=200,
                edgecolors='black', linewidth=2, label='Current Portfolio', zorder=5)

    plt.title(f'{portfolio_name} - Efficient Frontier (Monte Carlo)', fontsize=16, fontweight='bold')
    plt.xlabel('Volatility (Std Dev)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    plt.tight_layout()
    plt.savefig('out/out_optimise/efficient_frontier.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    # Save results
    optimal_df = pd.DataFrame({
        'ticker': tickers,
        'current_weight': current_weights,
        'optimal_weight': optimal_weights
    })

    stats_df = pd.DataFrame({
        'portfolio_type': ['Current', 'Optimal'],
        'expected_return': [current_return, optimal_return],
        'volatility': [current_std, optimal_std],
        'sharpe_ratio': [current_return / current_std, optimal_return / optimal_std]
    })

    optimal_df.to_csv('out/out_optimise/optimal_weights.csv', index=False)
    stats_df.to_csv('out/out_optimise/portfolio_stats.csv', index=False)

    print("Optimisation complete!")

    return optimal_df, stats_df


if __name__ == "__main__":
    optimal_weights, stats = main()