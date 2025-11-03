import json
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

    colors_green = ['#90EE90', '#52B788', '#2D6A4F', '#1B4332']
    cmap_green = LinearSegmentedColormap.from_list('green_gradient', colors_green)

    sharpe_min = results[2, :].min()
    sharpe_max = results[2, :].max()

    optimal_color_value = (optimal_sharpe - sharpe_min) / (sharpe_max - sharpe_min)
    optimal_color = cmap_green(optimal_color_value)

    current_color_value = (current_sharpe - sharpe_min) / (sharpe_max - sharpe_min)
    current_red = '#8B0000' if current_color_value < 0.5 else '#FF6B6B'

    # Visualise
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.weight'] = 'light'

    scatter = plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap=cmap_green,
                          alpha=0.4, s=10)
    cbar = plt.colorbar(scatter, label='Sharpe Ratio')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    plt.scatter(optimal_std, optimal_return, marker='o', color=optimal_color, s=200,
                edgecolors='white', linewidth=2, label='Optimal Portfolio', zorder=5)

    plt.scatter(current_std, current_return, marker='o', color=current_red, s=200,
                edgecolors='white', linewidth=2, label='Current Portfolio', zorder=5)

    plt.title(f'{portfolio_name} - Efficient Frontier (Monte Carlo)', fontsize=16, fontweight='light', color='white')
    plt.xlabel('Volatility (Std Dev)', fontsize=12, fontweight='light', color='white')
    plt.ylabel('Expected Return', fontsize=12, fontweight='light', color='white')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.2, linestyle='--', color='gray')

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.gca().tick_params(colors='white')

    plt.tight_layout()
    plt.savefig('out/out_optimise/efficient_frontier.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

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

    optimal_df.to_csv('out/out_optimise/optimal_weights.csv', index=False)
    stats_df.to_csv('out/out_optimise/portfolio_stats.csv', index=False)

    print("Optimisation complete!")

    return optimal_df, stats_df


if __name__ == "__main__":
    optimal_weights, stats = main()