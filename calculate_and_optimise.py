import json
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def main():
    os.makedirs('out/out_calculate', exist_ok=True)
    os.makedirs('out/out_optimise', exist_ok=True)
    os.makedirs('out/out_visualise', exist_ok=True)

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
            return None
    except Exception as e:
        print("Download unsuccessful!")
        return None

    try:
        market_prices = yf.download('^GSPC', start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
    except:
        print("Warning: Market data download unsuccessful!")
        market_prices = None

    returns = prices.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)

    calc_details = {
        'portfolio_name': portfolio_name,
        'tickers': ','.join(tickers),
        'weights': ','.join(map(str, weights)),
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'window': window,
        'trading_days': len(returns)
    }
    calc_in = pd.DataFrame([calc_details])

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

    correlation_matrix = returns.corr()
    corr = correlation_matrix.copy()

    calc_in.to_csv('out/out_calculate/calc_in.csv', index=False)
    vol.to_csv('out/out_calculate/vol.csv', index=False)
    corr.to_csv('out/out_calculate/corr.csv')

    # Monte Carlo Optimization
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    num_portfolios = 100000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        sim_weights = np.random.random(len(tickers))
        sim_weights /= np.sum(sim_weights)
        weights_record.append(sim_weights)
        portfolio_return = np.sum(sim_weights * mean_returns)
        portfolio_std = np.sqrt(np.dot(sim_weights.T, np.dot(cov_matrix, sim_weights)))
        sharpe_ratio = portfolio_return / portfolio_std
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe_ratio

    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]
    optimal_return = results[0, max_sharpe_idx]
    optimal_std = results[1, max_sharpe_idx]
    optimal_sharpe = results[2, max_sharpe_idx]

    current_return = np.sum(weights * mean_returns)
    current_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    current_sharpe = current_return / current_std

    historical_portfolio_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    historical_portfolio_vol = portfolio_returns.std() * np.sqrt(252)

    if market_prices is not None:
        market_returns = market_prices.pct_change().dropna()
        common_dates = portfolio_returns.index.intersection(market_returns.index)
        portfolio_returns_aligned = portfolio_returns.loc[common_dates].values.flatten()
        market_returns_aligned = market_returns.loc[common_dates].values.flatten()
        market_mean_return = np.mean(market_returns_aligned) * 252
        alpha = current_return - market_mean_return
        covariance_matrix = np.cov(portfolio_returns_aligned, market_returns_aligned)
        covariance = covariance_matrix[0, 1]
        market_variance = np.var(market_returns_aligned, ddof=1)
        beta = covariance / market_variance
    else:
        alpha = None
        beta = None

    return_distance = optimal_return - current_return
    volatility_distance = optimal_std - current_std
    sharpe_distance = optimal_sharpe - current_sharpe
    return_distance_pct = (return_distance / current_return) * 100
    volatility_distance_pct = (volatility_distance / current_std) * 100
    sharpe_distance_pct = (sharpe_distance / current_sharpe) * 100
    weights_distance = np.sum(np.abs(optimal_weights - weights))

    optimal_df = pd.DataFrame({
        'ticker': tickers,
        'current_weight': weights,
        'optimal_weight': optimal_weights,
        'weight_change': optimal_weights - weights,
        'weight_change_pct': ((optimal_weights - weights) / weights) * 100
    })

    stats_df = pd.DataFrame({
        'portfolio_type': ['Current', 'Optimal'],
        'expected_return': [current_return, optimal_return],
        'volatility': [current_std, optimal_std],
        'sharpe_ratio': [current_sharpe, optimal_sharpe],
        'historical_return': [historical_portfolio_return, np.nan],
        'historical_volatility': [historical_portfolio_vol, np.nan],
        'alpha': [alpha if alpha is not None else np.nan, np.nan],
        'beta': [beta if beta is not None else np.nan, np.nan]
    })

    results_df = pd.DataFrame({
        'returns': results[0, :],
        'volatility': results[1, :],
        'sharpe_ratio': results[2, :]
    })

    distance_metrics = pd.DataFrame({
        'metric': ['Expected Return Distance', 'Volatility Distance', 'Sharpe Ratio Distance',
                   'Return Distance (%)', 'Volatility Distance (%)', 'Sharpe Distance (%)',
                   'Total Weights Change (L1)', 'Alpha', 'Beta'],
        'value': [return_distance, volatility_distance, sharpe_distance, return_distance_pct,
                  volatility_distance_pct, sharpe_distance_pct, weights_distance,
                  alpha if alpha is not None else np.nan, beta if beta is not None else np.nan]
    })

    sharpe_min = results[2, :].min()
    sharpe_max = results[2, :].max()
    vol_min = results[1, :].min()
    vol_max = results[1, :].max()

    gradient_data = pd.DataFrame({
        'metric': ['Current Sharpe', 'Optimal Sharpe', 'Current Volatility', 'Optimal Volatility'],
        'raw_value': [current_sharpe, optimal_sharpe, current_std, optimal_std],
        'normalized_value': [
            (current_sharpe - sharpe_min) / (sharpe_max - sharpe_min),
            (optimal_sharpe - sharpe_min) / (sharpe_max - sharpe_min),
            (current_std - vol_min) / (vol_max - vol_min),
            (optimal_std - vol_min) / (vol_max - vol_min)
        ]
    })

    optimal_df.to_csv('out/out_optimise/optimal_weights.csv', index=False)
    stats_df.to_csv('out/out_optimise/portfolio_stats.csv', index=False)
    results_df.to_csv('out/out_optimise/simulation_results.csv', index=False)
    distance_metrics.to_csv('out/out_optimise/distance_metrics.csv', index=False)
    gradient_data.to_csv('out/out_optimise/gradient_data.csv', index=False)

    print("Calculation complete!")
    print("Optimisation complete!")
    print(f"\nCurrent Portfolio:")
    print(f"  Expected Return: {current_return:.2%}")
    print(f"  Volatility: {current_std:.2%}")
    print(f"  Sharpe Ratio: {current_sharpe:.4f}")
    print(f"\nOptimal Portfolio:")
    print(f"  Expected Return: {optimal_return:.2%}")
    print(f"  Volatility: {optimal_std:.2%}")
    print(f"  Sharpe Ratio: {optimal_sharpe:.4f}")
    print(f"\nImprovement Potential:")
    print(f"  Return: +{return_distance_pct:.2f}%")
    print(f"  Volatility: {volatility_distance_pct:+.2f}%")
    print(f"  Sharpe: +{sharpe_distance_pct:.2f}%")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(portfolio_name, vol, corr, stats_df, results_df, gradient_data, window)

    return calc_in, vol, corr, optimal_df, stats_df


def create_visualizations(portfolio_name, vol, corr, stats_df, results_df, gradient_data, window):
    vol['date'] = pd.to_datetime(vol['date'])
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.weight'] = 'light'

    # 1. Volatility
    plt.figure(figsize=(14, 7))
    plt.plot(vol['date'], vol['volatility'], linewidth=2, color='#52B788')
    plt.fill_between(vol['date'], vol['volatility'], alpha=0.4, color='#52B788')
    plt.title(f'Rolling {window}-Day Annualized Volatility', fontsize=14, fontweight='light', color='white')
    plt.xlabel('Date', fontsize=11, fontweight='light', color='white')
    plt.ylabel('Annualized Volatility', fontsize=11, fontweight='light', color='white')
    plt.grid(True, alpha=0.2, linestyle='--', color='gray')
    plt.ylim(bottom=0)
    plt.xlim(vol['date'].iloc[0], vol['date'].iloc[-1])
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    plt.gca().tick_params(colors='white')
    stats_text = f'{portfolio_name}\n\nCurrent: {vol["volatility"].iloc[-1]:.2%}\nMean: {vol["volatility"].mean():.2%}\nMax: {vol["volatility"].max():.2%}\nMin: {vol["volatility"].min():.2%}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
             fontweight='light', color='white',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, edgecolor='#52B788', linewidth=1.5))
    plt.tight_layout()
    plt.savefig('out/out_visualise/volatility.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    # 2. Correlation
    corr_display = corr.copy()
    np.fill_diagonal(corr_display.values, np.nan)
    cmap = LinearSegmentedColormap.from_list('green_transparent',
                                             [(0.0, 0.0, 0.0, 0.0), (0.32, 0.72, 0.53, 1.0), (0.0, 0.0, 0.0, 0.0)],
                                             N=256)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    mask_annot = np.eye(len(corr), dtype=bool)
    annot_data = corr.copy()
    annot_data[mask_annot] = np.nan
    sns.heatmap(corr_display, annot=annot_data, fmt='.2f', cmap=cmap, center=0, square=True, linewidths=1,
                linecolor='#2a2a2a', cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, ax=ax,
                annot_kws={'color': 'white', 'fontsize': 10})
    plt.title(f'{portfolio_name} - Correlation Matrix', fontsize=16, fontweight='light', color='white', pad=20)
    ax.tick_params(colors='white')
    plt.setp(ax.get_xticklabels(), color='white')
    plt.setp(ax.get_yticklabels(), color='white')
    plt.tight_layout()
    plt.savefig('out/out_visualise/correlation.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    # 3. Efficient Frontier
    current = stats_df[stats_df['portfolio_type'] == 'Current'].iloc[0]
    optimal = stats_df[stats_df['portfolio_type'] == 'Optimal'].iloc[0]
    cmap_frontier = LinearSegmentedColormap.from_list('green', ['#90EE90', '#52B788', '#2D6A4F', '#1B4332'])
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    scatter = ax.scatter(results_df['volatility'], results_df['returns'], c=results_df['sharpe_ratio'],
                         cmap=cmap_frontier, alpha=0.6, s=10, edgecolors='none')
    ax.scatter(current['volatility'], current['expected_return'], color='#FF6B6B', s=500, marker='o',
               edgecolors='white', linewidths=2, label='Current', zorder=5)
    ax.scatter(optimal['volatility'], optimal['expected_return'], color='#52B788', s=500, marker='*',
               edgecolors='white', linewidths=2, label='Optimal', zorder=5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20, color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    ax.set_xlabel('Volatility', fontsize=12, color='white', fontweight='light')
    ax.set_ylabel('Expected Return', fontsize=12, color='white', fontweight='light')
    ax.set_title(f'{portfolio_name} - Efficient Frontier', fontsize=16, color='white', fontweight='light', pad=20)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, linestyle='--', color='gray')
    legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('#1a1a1a')
    legend.get_frame().set_edgecolor('#52B788')
    for text in legend.get_texts():
        text.set_color('white')
    plt.tight_layout()
    plt.savefig('out/out_visualise/efficient_frontier.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    # 4. Gradient Bars
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    sharpe_data = gradient_data[gradient_data['metric'].str.contains('Sharpe')]
    vol_data = gradient_data[gradient_data['metric'].str.contains('Volatility')]
    y_pos = [0, 1, 3, 4]
    labels = sharpe_data['metric'].tolist() + vol_data['metric'].tolist()
    values = sharpe_data['normalized_value'].tolist() + vol_data['normalized_value'].tolist()
    raw_values = sharpe_data['raw_value'].tolist() + vol_data['raw_value'].tolist()
    colors_list = ['#FF6B6B', '#52B788', '#FF6B6B', '#52B788']
    bars = ax.barh(y_pos, values, color=colors_list, height=0.6, edgecolor='white', linewidth=1.5)
    for i, (bar, raw_val) in enumerate(zip(bars, raw_values)):
        label_text = f'{raw_val:.4f}' if 'Sharpe' in labels[i] else f'{raw_val:.2%}'
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, label_text, va='center', ha='left',
                color='white', fontsize=10, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, color='white')
    ax.set_xlabel('Normalized Value', fontsize=12, color='white', fontweight='light')
    ax.set_title(f'{portfolio_name} - Current vs Optimal', fontsize=16, color='white', fontweight='light', pad=20)
    ax.set_xlim(0, 1.15)
    ax.tick_params(colors='white')
    ax.grid(True, axis='x', alpha=0.2, linestyle='--', color='gray')
    plt.tight_layout()
    plt.savefig('out/out_visualise/gradient_comparison.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print("Visualizations created!")


if __name__ == "__main__":
    calc_in, vol, corr, optimal_df, stats_df = main()