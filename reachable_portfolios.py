import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import yfinance as yf
from datetime import datetime


def generate_buy_only_portfolios(current_weights, tickers, num_portfolios=100000):
    """Generate diverse portfolios via Monte Carlo"""
    portfolios = []
    budget = 0.10  # Fixed to exactly 10%

    np.random.seed(42)

    for i in range(num_portfolios):
        num_buys = np.random.choice([1, 2, 3, 4, 5])
        num_buys = min(num_buys, len(tickers))
        buy_indices = np.random.choice(len(tickers), size=num_buys, replace=False)

        if num_buys == 1:
            buy_amounts = [budget]
        else:
            # Varied allocation strategies
            if i % 4 == 0:
                buy_amounts = [budget / num_buys] * num_buys
            elif i % 4 == 1:
                main_pct = np.random.uniform(0.70, 0.85)
                main_amount = budget * main_pct
                remaining = budget - main_amount
                other_amounts = np.random.dirichlet(np.ones(num_buys - 1)) * remaining
                buy_amounts = np.append([main_amount], other_amounts)
                np.random.shuffle(buy_amounts)
            elif i % 4 == 2:
                main_pct = np.random.uniform(0.40, 0.60)
                main_amount = budget * main_pct
                remaining = budget - main_amount
                other_amounts = np.random.dirichlet(np.ones(num_buys - 1)) * remaining
                buy_amounts = np.append([main_amount], other_amounts)
                np.random.shuffle(buy_amounts)
            else:
                splits = np.random.dirichlet(np.ones(num_buys) * 2)
                buy_amounts = splits * budget

        new_weights = current_weights.copy()
        total_before = np.sum(new_weights)

        for idx, amount in zip(buy_indices, buy_amounts):
            new_weights[idx] += amount

        new_weights = new_weights / np.sum(new_weights)

        # Create trade instructions
        trades = []
        buy_stocks = []
        for idx, amount in zip(buy_indices, buy_amounts):
            pct = (amount / total_before) * 100
            trades.append(f"Buy {tickers[idx]} {pct:.1f}%")
            buy_stocks.append(tickers[idx])

        # Create unique signature: sorted list of (stock, rounded_amount) to detect duplicates
        signature_parts = []
        for idx, amount in zip(buy_indices, buy_amounts):
            pct = (amount / total_before) * 100
            # Round to 0.5% precision to catch near-duplicates
            rounded_pct = round(pct * 2) / 2  # Round to nearest 0.5%
            if rounded_pct > 0:  # Only include non-zero amounts
                signature_parts.append(f"{tickers[idx]}:{rounded_pct:.1f}")

        signature = '|'.join(sorted(signature_parts))

        portfolios.append({
            'weights': new_weights,
            'trades': '; '.join(trades),
            'num_operations': num_buys,
            'buy_tickers': ','.join(buy_stocks),
            'signature': signature  # For deduplication
        })

    return portfolios


def main():
    os.makedirs('out/out_reachable', exist_ok=True)

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio = data['portfolio']
    specs = data['specs']

    tickers = portfolio['tickers']
    current_weights = np.array(portfolio['weights'])
    portfolio_name = portfolio['name']

    stats_df = pd.read_csv('out/out_optimise/portfolio_stats.csv')
    current = stats_df[stats_df['portfolio_type'] == 'Current'].iloc[0]
    current_return, current_std, current_sharpe = current['expected_return'], current['volatility'], current[
        'sharpe_ratio']

    start_date = datetime.strptime(specs['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(specs['end_date'], '%Y-%m-%d')

    prices = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices['Adj Close']

    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    print("Generating 100K buy-only portfolios...")
    generated = generate_buy_only_portfolios(current_weights, tickers, 100000)

    print("Calculating metrics...")
    results = []
    seen_signatures = set()

    for p in generated:
        # Skip if we've seen this exact portfolio before
        if p['signature'] in seen_signatures:
            continue
        seen_signatures.add(p['signature'])

        ret = np.sum(p['weights'] * mean_returns.values)
        vol = np.sqrt(np.dot(p['weights'].T, np.dot(cov_matrix.values, p['weights'])))
        sharpe = ret / vol if vol > 0 else 0
        results.append({
            'returns': ret, 'volatility': vol, 'sharpe_ratio': sharpe,
            'trades': p['trades'], 'num_operations': p['num_operations'],
            'return_improvement': ((ret - current_return) / current_return * 100),
            'volatility_change': ((vol - current_std) / current_std * 100),
            'sharpe_improvement': ((sharpe - current_sharpe) / current_sharpe * 100),
            'signature': p['signature']
        })

    results_df = pd.DataFrame(results)
    better = results_df[results_df['sharpe_ratio'] > current_sharpe].copy()

    print(f"Generated {len(results_df):,} unique portfolios, {len(better):,} better than current")

    # Get top 10 for each operation count
    top_by_ops = []
    for ops in range(1, 6):
        ops_data = better[better['num_operations'] == ops].copy()
        if len(ops_data) > 0:
            # Simply take top 10 by Sharpe - duplicates already removed
            top_10 = ops_data.nlargest(min(10, len(ops_data)), 'sharpe_ratio')
            top_by_ops.append(top_10)

    all_recommendations = pd.concat(top_by_ops, ignore_index=True) if top_by_ops else pd.DataFrame()

    # Save
    all_recommendations.to_csv('out/out_reachable/reachable_portfolios.csv', index=False)

    summary = pd.DataFrame({
        'metric': ['Total Generated', 'Better Sharpe'] + [f'{i} Ops' for i in range(1, 6)],
        'value': [len(results_df), len(better)] + [len(better[better['num_operations'] == i]) for i in range(1, 6)]
    })
    summary.to_csv('out/out_reachable/summary_stats.csv', index=False)

    # Visualize
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#1a1a1a')
    cmap = LinearSegmentedColormap.from_list('green', ['#90EE90', '#52B788', '#2D6A4F', '#1B4332'])

    for ax in axes.flat:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.set_xlabel('Volatility', color='white')
        ax.set_ylabel('Return', color='white')
        ax.grid(True, alpha=0.2, color='gray')

    # Plot 1: All
    axes[0, 0].scatter(results_df['volatility'], results_df['returns'], c=results_df['sharpe_ratio'], cmap=cmap,
                       alpha=0.3, s=5)
    axes[0, 0].scatter(current_std, current_return, marker='o', color='#FF6B6B', s=300, edgecolors='white', linewidth=2,
                       label='Current', zorder=5)
    axes[0, 0].set_title('All Portfolios (100K)', fontsize=14, color='white', fontweight='light')
    axes[0, 0].legend()

    # Plot 2-3: Categories
    for idx, (mask, title) in enumerate([
        ((better['returns'] > current_return) & (better['volatility'] <= current_std * 1.05), 'Higher Return'),
        ((better['volatility'] < current_std) & (better['returns'] >= current_return * 0.95), 'Lower Volatility')
    ]):
        ax = axes[0, 1] if idx == 0 else axes[1, 0]
        cat_data = better[mask]
        if len(cat_data) > 0:
            ax.scatter(cat_data['volatility'], cat_data['returns'], c=cat_data['sharpe_ratio'], cmap=cmap, s=30,
                       alpha=0.6)
            top = cat_data.nlargest(10, 'sharpe_ratio')
            ax.scatter(top['volatility'], top['returns'], marker='*', s=300, color='gold', edgecolors='white',
                       linewidth=2, label='Top 10', zorder=5)
        ax.scatter(current_std, current_return, marker='o', color='#FF6B6B', s=300, edgecolors='white', linewidth=2,
                   label='Current', zorder=5)
        ax.set_title(title, fontsize=14, color='white', fontweight='light')
        ax.legend()

    # Plot 4: By operations
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#90EE90', '#52B788', '#2D6A4F', '#1B4332', '#0D3B1F']
    for ops in range(1, 6):
        ops_data = all_recommendations[all_recommendations['num_operations'] == ops]
        if len(ops_data) > 0:
            axes[1, 1].scatter(ops_data['volatility'], ops_data['returns'], color=colors[ops - 1], s=100,
                               marker=markers[ops - 1], label=f'{ops} Op{"s" if ops > 1 else ""}',
                               edgecolors='white', linewidths=1.5, alpha=0.8)
    axes[1, 1].scatter(current_std, current_return, marker='o', color='#FF6B6B', s=300, edgecolors='white', linewidth=2,
                       label='Current', zorder=6)
    axes[1, 1].set_title('Top 10 by Operations', fontsize=14, color='white', fontweight='light')
    axes[1, 1].legend(fontsize=9)

    plt.suptitle(f'{portfolio_name} - Buy-Only Improvements (10% Budget, 1-5 Ops)', fontsize=18, color='#52B788',
                 fontweight='light', y=0.995)
    plt.tight_layout()
    plt.savefig('out/out_reachable/reachable_improvements.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print("Complete!")
    for ops in range(1, 6):
        print(f"  {ops} op{'s' if ops > 1 else ''}: {len(better[better['num_operations'] == ops]):,}")

    return all_recommendations, summary


if __name__ == "__main__":
    recommendations, summary = main()