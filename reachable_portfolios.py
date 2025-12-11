import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def generate_buy_only_portfolios(current_weights, tickers, num_portfolios=50000):
    """Generate portfolios by only adding (buying) with 10% budget"""
    portfolios = []
    budget = 0.10  # 10% of portfolio to invest

    for _ in range(num_portfolios):
        # Randomly decide: 1 or 2 stocks to buy
        num_buys = np.random.choice([1, 2])

        # Select which stocks to buy
        buy_indices = np.random.choice(len(tickers), size=num_buys, replace=False)

        # Allocate budget across selected stocks
        if num_buys == 1:
            buy_amounts = [budget]
        else:
            # Random split of budget between 2 stocks
            split = np.random.uniform(0.2, 0.8)  # Ensure meaningful allocation
            buy_amounts = [budget * split, budget * (1 - split)]

        # Create new weights
        new_weights = current_weights.copy()
        total_before = np.sum(new_weights)

        for idx, amount in zip(buy_indices, buy_amounts):
            new_weights[idx] += amount

        # Normalize to ensure weights sum to 1
        new_weights = new_weights / np.sum(new_weights)

        # Create trade instructions
        trades = []
        for idx, amount in zip(buy_indices, buy_amounts):
            # Calculate percentage of total portfolio
            pct = (amount / total_before) * 100
            trades.append(f"Buy {tickers[idx]} {pct:.1f}%")

        portfolios.append({
            'weights': new_weights,
            'trades': '; '.join(trades),
            'num_operations': num_buys,
            'buy_tickers': ','.join([tickers[i] for i in buy_indices])
        })

    return portfolios


def calculate_portfolio_metrics(weights, mean_returns, cov_matrix):
    """Calculate return, volatility, and Sharpe ratio"""
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
    return portfolio_return, portfolio_std, sharpe_ratio


def main():
    os.makedirs('out/out_reachable', exist_ok=True)

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio = data['portfolio']

    tickers = portfolio['tickers']
    current_weights = np.array(portfolio['weights'])
    portfolio_name = portfolio['name']

    # Load optimization data to get mean returns and covariance
    stats_df = pd.read_csv('out/out_optimise/portfolio_stats.csv')

    # Need to recalculate from raw data
    from datetime import datetime
    import yfinance as yf

    specs = data['specs']
    start_date = datetime.strptime(specs['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(specs['end_date'], '%Y-%m-%d')

    try:
        prices = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices['Adj Close']
        else:
            prices = prices[['Adj Close']] if 'Adj Close' in prices.columns else prices

        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
    except:
        print("Error downloading data for metrics calculation")
        return None

    current_return = stats_df[stats_df['portfolio_type'] == 'Current']['expected_return'].iloc[0]
    current_std = stats_df[stats_df['portfolio_type'] == 'Current']['volatility'].iloc[0]
    current_sharpe = stats_df[stats_df['portfolio_type'] == 'Current']['sharpe_ratio'].iloc[0]

    print("Generating buy-only portfolios...")
    generated_portfolios = generate_buy_only_portfolios(current_weights, tickers, num_portfolios=50000)

    print("Calculating metrics for all portfolios...")
    results = []
    for p in generated_portfolios:
        ret, vol, sharpe = calculate_portfolio_metrics(p['weights'], mean_returns.values, cov_matrix.values)
        results.append({
            'returns': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'trades': p['trades'],
            'num_operations': p['num_operations'],
            'buy_tickers': p['buy_tickers'],
            'return_improvement': ((ret - current_return) / current_return * 100) if current_return != 0 else 0,
            'volatility_change': ((vol - current_std) / current_std * 100) if current_std != 0 else 0,
            'sharpe_improvement': ((sharpe - current_sharpe) / current_sharpe * 100) if current_sharpe != 0 else 0
        })

    results_df = pd.DataFrame(results)

    # Filter for improvements
    better_portfolios = results_df[results_df['sharpe_ratio'] > current_sharpe].copy()

    print(f"Found {len(better_portfolios)} portfolios with better Sharpe ratio")

    # Category 1: Higher return with similar/lower volatility
    higher_return = better_portfolios[
        (better_portfolios['returns'] > current_return) &
        (better_portfolios['volatility'] <= current_std * 1.05)
        ].copy()
    higher_return['category'] = 'Higher Return'

    # Category 2: Lower volatility with similar/higher return
    lower_vol = better_portfolios[
        (better_portfolios['volatility'] < current_std) &
        (better_portfolios['returns'] >= current_return * 0.95)
        ].copy()
    lower_vol['category'] = 'Lower Volatility'

    # Category 3: Best Sharpe improvements
    better_sharpe = better_portfolios.copy()
    better_sharpe['category'] = 'Better Sharpe'

    # Get top portfolios from each category
    top_higher_return = higher_return.nlargest(10, 'returns') if len(higher_return) > 0 else pd.DataFrame()
    top_lower_vol = lower_vol.nsmallest(10, 'volatility') if len(lower_vol) > 0 else pd.DataFrame()
    top_better_sharpe = better_sharpe.nlargest(10, 'sharpe_ratio')

    # Combine and remove duplicates
    all_recommendations = pd.concat([top_higher_return, top_lower_vol, top_better_sharpe], ignore_index=True)
    all_recommendations = all_recommendations.drop_duplicates(subset=['returns', 'volatility', 'sharpe_ratio'])

    # Separate by number of operations
    one_op = all_recommendations[all_recommendations['num_operations'] == 1].copy()
    two_op = all_recommendations[all_recommendations['num_operations'] == 2].copy()

    # Summary statistics
    summary_stats = pd.DataFrame({
        'metric': [
            'Total Portfolios Generated',
            'Portfolios with Better Sharpe',
            'Higher Return Options',
            'Lower Volatility Options',
            'Single Operation Portfolios',
            'Two Operation Portfolios',
            'Best Return Improvement',
            'Best Volatility Reduction',
            'Best Sharpe Improvement'
        ],
        'value': [
            len(results_df),
            len(better_portfolios),
            len(higher_return),
            len(lower_vol),
            len(one_op),
            len(two_op),
            f"{all_recommendations['return_improvement'].max():.2f}%" if len(all_recommendations) > 0 else "N/A",
            f"{all_recommendations['volatility_change'].min():.2f}%" if len(all_recommendations) > 0 else "N/A",
            f"{all_recommendations['sharpe_improvement'].max():.2f}%" if len(all_recommendations) > 0 else "N/A"
        ]
    })

    # Save outputs
    all_recommendations.to_csv('out/out_reachable/reachable_portfolios.csv', index=False)
    summary_stats.to_csv('out/out_reachable/summary_stats.csv', index=False)
    results_df.to_csv('out/out_reachable/all_generated_portfolios.csv', index=False)

    # Create visualization
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#1a1a1a')

    colors_green = ['#90EE90', '#52B788', '#2D6A4F', '#1B4332']
    cmap_green = LinearSegmentedColormap.from_list('green_gradient', colors_green)

    # Plot 1: All generated portfolios
    ax1 = axes[0, 0]
    ax1.set_facecolor('#1a1a1a')

    scatter1 = ax1.scatter(results_df['volatility'], results_df['returns'],
                           c=results_df['sharpe_ratio'], cmap=cmap_green, alpha=0.3, s=10)
    ax1.scatter(current_std, current_return, marker='o', color='#FF6B6B', s=300,
                edgecolors='white', linewidth=2, label='Current', zorder=5)

    plt.colorbar(scatter1, ax=ax1, label='Sharpe Ratio')
    ax1.set_title('All Buy-Only Portfolios (10% Budget)', fontsize=14, fontweight='light', color='white')
    ax1.set_xlabel('Volatility', fontsize=11, color='white')
    ax1.set_ylabel('Expected Return', fontsize=11, color='white')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, color='gray')
    ax1.tick_params(colors='white')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    # Plot 2: Higher Return
    ax2 = axes[0, 1]
    ax2.set_facecolor('#1a1a1a')

    if len(higher_return) > 0:
        ax2.scatter(results_df['volatility'], results_df['returns'], alpha=0.1, s=5, color='gray')
        scatter2 = ax2.scatter(higher_return['volatility'], higher_return['returns'],
                               c=higher_return['sharpe_ratio'], cmap=cmap_green, s=50, alpha=0.7)
        if len(top_higher_return) > 0:
            ax2.scatter(top_higher_return['volatility'], top_higher_return['returns'],
                        marker='*', s=400, color='gold', edgecolors='white', linewidth=2, label='Top 10', zorder=5)

    ax2.scatter(current_std, current_return, marker='o', color='#FF6B6B', s=300,
                edgecolors='white', linewidth=2, label='Current', zorder=5)
    ax2.set_title('Higher Return Options', fontsize=14, fontweight='light', color='white')
    ax2.set_xlabel('Volatility', fontsize=11, color='white')
    ax2.set_ylabel('Expected Return', fontsize=11, color='white')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2, color='gray')
    ax2.tick_params(colors='white')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    # Plot 3: Lower Volatility
    ax3 = axes[1, 0]
    ax3.set_facecolor('#1a1a1a')

    if len(lower_vol) > 0:
        ax3.scatter(results_df['volatility'], results_df['returns'], alpha=0.1, s=5, color='gray')
        scatter3 = ax3.scatter(lower_vol['volatility'], lower_vol['returns'],
                               c=lower_vol['sharpe_ratio'], cmap=cmap_green, s=50, alpha=0.7)
        if len(top_lower_vol) > 0:
            ax3.scatter(top_lower_vol['volatility'], top_lower_vol['returns'],
                        marker='*', s=400, color='gold', edgecolors='white', linewidth=2, label='Top 10', zorder=5)

    ax3.scatter(current_std, current_return, marker='o', color='#FF6B6B', s=300,
                edgecolors='white', linewidth=2, label='Current', zorder=5)
    ax3.set_title('Lower Volatility Options', fontsize=14, fontweight='light', color='white')
    ax3.set_xlabel('Volatility', fontsize=11, color='white')
    ax3.set_ylabel('Expected Return', fontsize=11, color='white')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.2, color='gray')
    ax3.tick_params(colors='white')
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    # Plot 4: By number of operations
    ax4 = axes[1, 1]
    ax4.set_facecolor('#1a1a1a')

    if len(one_op) > 0:
        ax4.scatter(one_op['volatility'], one_op['returns'],
                    c=one_op['sharpe_ratio'], cmap=cmap_green, s=100, alpha=0.7,
                    marker='o', label='1 Operation', edgecolors='white', linewidths=1)
    if len(two_op) > 0:
        ax4.scatter(two_op['volatility'], two_op['returns'],
                    c=two_op['sharpe_ratio'], cmap=cmap_green, s=100, alpha=0.7,
                    marker='s', label='2 Operations', edgecolors='white', linewidths=1)

    ax4.scatter(current_std, current_return, marker='o', color='#FF6B6B', s=300,
                edgecolors='white', linewidth=2, label='Current', zorder=5)
    ax4.set_title('By Number of Operations', fontsize=14, fontweight='light', color='white')
    ax4.set_xlabel('Volatility', fontsize=11, color='white')
    ax4.set_ylabel('Expected Return', fontsize=11, color='white')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.2, color='gray')
    ax4.tick_params(colors='white')
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    plt.suptitle(f'{portfolio_name} - Buy-Only Portfolio Improvements (10% Budget, Max 2 Operations)',
                 fontsize=18, fontweight='light', color='#52B788', y=0.995)

    plt.tight_layout()
    plt.savefig('out/out_reachable/reachable_improvements.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print("Reachable improvements analysis complete!")
    print(f"Total portfolios generated: {len(results_df):,}")
    print(f"Better Sharpe portfolios: {len(better_portfolios):,}")
    print(f"Single operation: {len(one_op)}")
    print(f"Two operations: {len(two_op)}")
    print(f"Top recommendations: {len(all_recommendations)}")

    return all_recommendations, summary_stats


if __name__ == "__main__":
    recommendations, summary = main()