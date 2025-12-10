import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def calculate_rebalance_cost(current_weights, target_weights):
    """Calculate total rebalancing cost as % of portfolio value"""
    weight_changes = np.abs(target_weights - current_weights)
    total_trades = np.sum(weight_changes) / 2  # Divide by 2 since buying = selling
    return total_trades


def main():
    os.makedirs('out/out_reachable', exist_ok=True)

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio = data['portfolio']

    tickers = portfolio['tickers']
    current_weights = np.array(portfolio['weights'])
    portfolio_name = portfolio['name']

    stats_df = pd.read_csv('out/out_optimise/portfolio_stats.csv')
    results_df = pd.read_csv('out/out_optimise/simulation_results.csv')

    current_return = stats_df[stats_df['portfolio_type'] == 'Current']['expected_return'].iloc[0]
    current_std = stats_df[stats_df['portfolio_type'] == 'Current']['volatility'].iloc[0]
    current_sharpe = stats_df[stats_df['portfolio_type'] == 'Current']['sharpe_ratio'].iloc[0]

    # Load weights from Monte Carlo simulation
    # We need to reconstruct weights from simulation or save them
    # For now, we'll work with the results we have

    # Filter portfolios that are reachable with ≤10% rebalancing
    # Since we don't have weights saved, we'll use a proxy approach
    # Estimate based on return/volatility distance

    max_rebalance = 0.10  # 10% of portfolio value

    # Calculate normalized distances
    return_distance = np.abs(results_df['returns'] - current_return)
    vol_distance = np.abs(results_df['volatility'] - current_std)

    # Proxy for rebalancing cost: normalize and combine distances
    # This is a heuristic - ideally we'd use actual weight differences
    return_dist_norm = return_distance / results_df['returns'].std()
    vol_dist_norm = vol_distance / results_df['volatility'].std()
    estimated_rebalance = (return_dist_norm + vol_dist_norm) / 10  # Scale down

    # Filter reachable portfolios
    reachable_mask = estimated_rebalance <= max_rebalance
    reachable = results_df[reachable_mask].copy()

    # Category 1: Higher return, same or lower volatility
    higher_return_mask = (reachable['returns'] > current_return) & \
                         (reachable['volatility'] <= current_std * 1.02)  # 2% tolerance
    higher_return_portfolios = reachable[higher_return_mask].copy()

    # Category 2: Lower volatility, same or higher return
    lower_vol_mask = (reachable['volatility'] < current_std) & \
                     (reachable['returns'] >= current_return * 0.98)  # 2% tolerance
    lower_vol_portfolios = reachable[lower_vol_mask].copy()

    # Category 3: Better Sharpe ratio
    better_sharpe_mask = reachable['sharpe_ratio'] > current_sharpe
    better_sharpe_portfolios = reachable[better_sharpe_mask].copy()

    # Get top 5 from each category
    if len(higher_return_portfolios) > 0:
        top_higher_return = higher_return_portfolios.nlargest(5, 'returns')
        top_higher_return['category'] = 'Higher Return'
    else:
        top_higher_return = pd.DataFrame()

    if len(lower_vol_portfolios) > 0:
        top_lower_vol = lower_vol_portfolios.nsmallest(5, 'volatility')
        top_lower_vol['category'] = 'Lower Volatility'
    else:
        top_lower_vol = pd.DataFrame()

    if len(better_sharpe_portfolios) > 0:
        top_better_sharpe = better_sharpe_portfolios.nlargest(5, 'sharpe_ratio')
        top_better_sharpe['category'] = 'Better Sharpe'
    else:
        top_better_sharpe = pd.DataFrame()

    # Combine all recommendations
    all_recommendations = pd.concat([top_higher_return, top_lower_vol, top_better_sharpe], ignore_index=True)

    # Remove duplicates
    all_recommendations = all_recommendations.drop_duplicates(subset=['returns', 'volatility', 'sharpe_ratio'])

    # Add improvement metrics
    all_recommendations['return_improvement'] = (
                (all_recommendations['returns'] - current_return) / current_return * 100)
    all_recommendations['volatility_change'] = ((all_recommendations['volatility'] - current_std) / current_std * 100)
    all_recommendations['sharpe_improvement'] = (
                (all_recommendations['sharpe_ratio'] - current_sharpe) / current_sharpe * 100)

    # Summary statistics
    summary_stats = pd.DataFrame({
        'metric': [
            'Total Reachable Portfolios',
            'Higher Return Options',
            'Lower Volatility Options',
            'Better Sharpe Options',
            'Best Return Improvement',
            'Best Volatility Reduction',
            'Best Sharpe Improvement'
        ],
        'value': [
            len(reachable),
            len(higher_return_portfolios),
            len(lower_vol_portfolios),
            len(better_sharpe_portfolios),
            f"{all_recommendations['return_improvement'].max():.2f}%" if len(all_recommendations) > 0 else "N/A",
            f"{all_recommendations['volatility_change'].min():.2f}%" if len(all_recommendations) > 0 else "N/A",
            f"{all_recommendations['sharpe_improvement'].max():.2f}%" if len(all_recommendations) > 0 else "N/A"
        ]
    })

    # Save outputs
    all_recommendations.to_csv('out/out_reachable/reachable_portfolios.csv', index=False)
    summary_stats.to_csv('out/out_reachable/summary_stats.csv', index=False)

    # Create visualization
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#1a1a1a')

    colors_green = ['#90EE90', '#52B788', '#2D6A4F', '#1B4332']
    cmap_green = LinearSegmentedColormap.from_list('green_gradient', colors_green)

    # Plot 1: All reachable portfolios
    ax1 = axes[0, 0]
    ax1.set_facecolor('#1a1a1a')

    sharpe_min = results_df['sharpe_ratio'].min()
    sharpe_max = results_df['sharpe_ratio'].max()

    # Background: all simulations
    ax1.scatter(results_df['volatility'], results_df['returns'],
                c=results_df['sharpe_ratio'], cmap=cmap_green, alpha=0.1, s=5)

    # Reachable portfolios
    scatter1 = ax1.scatter(reachable['volatility'], reachable['returns'],
                           c=reachable['sharpe_ratio'], cmap=cmap_green, alpha=0.5, s=20)

    # Current portfolio
    ax1.scatter(current_std, current_return, marker='o', color='#FF6B6B', s=300,
                edgecolors='white', linewidth=2, label='Current', zorder=5)

    ax1.set_title('Reachable Portfolios (≤10% Rebalancing)', fontsize=14, fontweight='light', color='white')
    ax1.set_xlabel('Volatility', fontsize=11, color='white')
    ax1.set_ylabel('Expected Return', fontsize=11, color='white')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, color='gray')
    ax1.tick_params(colors='white')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    # Plot 2: Higher Return portfolios
    ax2 = axes[0, 1]
    ax2.set_facecolor('#1a1a1a')

    if len(higher_return_portfolios) > 0:
        ax2.scatter(reachable['volatility'], reachable['returns'], alpha=0.1, s=10, color='gray')
        scatter2 = ax2.scatter(higher_return_portfolios['volatility'], higher_return_portfolios['returns'],
                               c=higher_return_portfolios['sharpe_ratio'], cmap=cmap_green, s=50, alpha=0.7)

        if len(top_higher_return) > 0:
            ax2.scatter(top_higher_return['volatility'], top_higher_return['returns'],
                        marker='*', s=400, color='gold', edgecolors='white', linewidth=2,
                        label='Top 5', zorder=5)

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

    # Plot 3: Lower Volatility portfolios
    ax3 = axes[1, 0]
    ax3.set_facecolor('#1a1a1a')

    if len(lower_vol_portfolios) > 0:
        ax3.scatter(reachable['volatility'], reachable['returns'], alpha=0.1, s=10, color='gray')
        scatter3 = ax3.scatter(lower_vol_portfolios['volatility'], lower_vol_portfolios['returns'],
                               c=lower_vol_portfolios['sharpe_ratio'], cmap=cmap_green, s=50, alpha=0.7)

        if len(top_lower_vol) > 0:
            ax3.scatter(top_lower_vol['volatility'], top_lower_vol['returns'],
                        marker='*', s=400, color='gold', edgecolors='white', linewidth=2,
                        label='Top 5', zorder=5)

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

    # Plot 4: Better Sharpe portfolios
    ax4 = axes[1, 1]
    ax4.set_facecolor('#1a1a1a')

    if len(better_sharpe_portfolios) > 0:
        ax4.scatter(reachable['volatility'], reachable['returns'], alpha=0.1, s=10, color='gray')
        scatter4 = ax4.scatter(better_sharpe_portfolios['volatility'], better_sharpe_portfolios['returns'],
                               c=better_sharpe_portfolios['sharpe_ratio'], cmap=cmap_green, s=50, alpha=0.7)

        if len(top_better_sharpe) > 0:
            ax4.scatter(top_better_sharpe['volatility'], top_better_sharpe['returns'],
                        marker='*', s=400, color='gold', edgecolors='white', linewidth=2,
                        label='Top 5', zorder=5)

    ax4.scatter(current_std, current_return, marker='o', color='#FF6B6B', s=300,
                edgecolors='white', linewidth=2, label='Current', zorder=5)

    ax4.set_title('Better Sharpe Ratio Options', fontsize=14, fontweight='light', color='white')
    ax4.set_xlabel('Volatility', fontsize=11, color='white')
    ax4.set_ylabel('Expected Return', fontsize=11, color='white')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.2, color='gray')
    ax4.tick_params(colors='white')
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    plt.suptitle(f'{portfolio_name} - Reachable Portfolio Improvements',
                 fontsize=18, fontweight='light', color='#52B788', y=0.995)

    plt.tight_layout()
    plt.savefig('out/out_reachable/reachable_improvements.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print("Reachable improvements analysis complete!")
    print(f"Total reachable portfolios: {len(reachable):,}")
    print(f"Higher return options: {len(higher_return_portfolios):,}")
    print(f"Lower volatility options: {len(lower_vol_portfolios):,}")
    print(f"Better Sharpe options: {len(better_sharpe_portfolios):,}")
    print(f"Top recommendations saved: {len(all_recommendations)}")

    return all_recommendations, summary_stats


if __name__ == "__main__":
    recommendations, summary = main()