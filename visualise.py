import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import json
import os

def main():
    if not os.path.exists('out/out_calculate'):
        print("Error: 'out/out_calculate' folder not found. Run main script first.")
        return

    if not os.path.exists('out/out_optimise'):
        print("Error: 'out/out_optimise' folder not found. Run optimise script first.")
        return

    os.makedirs('out/out_visualise', exist_ok=True)

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio_name = data['portfolio']['name']

    calc_in = pd.read_csv('out/out_calculate/calc_in.csv')
    vol = pd.read_csv('out/out_calculate/vol.csv')
    corr = pd.read_csv('out/out_calculate/corr.csv', index_col=0)

    stats_df = pd.read_csv('out/out_optimise/portfolio_stats.csv')
    results_df = pd.read_csv('out/out_optimise/simulation_results.csv')

    vol['date'] = pd.to_datetime(vol['date'])

    portfolio_name = calc_in['portfolio_name'].iloc[0]
    window = calc_in['window'].iloc[0]

    # Plot volatility
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 7))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.weight'] = 'light'

    vol['volatility'] = pd.to_numeric(vol['volatility'], errors='coerce')
    vol['date'] = pd.to_datetime(vol['date'], errors='coerce')
    vol = vol.dropna(subset=['date', 'volatility'])

    if vol.empty:
        print("Warning: No valid volatility data available for plotting.")
    else:
        vol = vol.sort_values(by='date')

        plt.plot(vol['date'], vol['volatility'], linewidth=2, color='#52B788')
        plt.fill_between(vol['date'], vol['volatility'], alpha=0.4, color='#52B788')

        plt.title(f'Rolling {window}-Day Annualized Volatility', fontsize=14, fontweight='light', color='white')
        plt.xlabel('Date', fontsize=11, fontweight='light', color='white')
        plt.ylabel('Annualized Volatility', fontsize=11, fontweight='light', color='white')
        plt.grid(True, alpha=0.2, linestyle='--', color='gray')

        plt.ylim(bottom=0)
        plt.xlim(vol['date'].iloc[0], vol['date'].iloc[-1])

        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        plt.gca().tick_params(colors='white')

        mean_vol = vol['volatility'].mean()
        max_vol = vol['volatility'].max()
        min_vol = vol['volatility'].min()
        current_vol = vol['volatility'].iloc[-1]

        stats_text = f'{portfolio_name}\n\nCurrent: {current_vol:.2%}\nMean: {mean_vol:.2%}\nMax: {max_vol:.2%}\nMin: {min_vol:.2%}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='top', fontweight='light', color='white',
                 bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, edgecolor='#52B788', linewidth=1.5))

        plt.tight_layout()
        plt.savefig('out/out_visualise/volatility.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()

    # Correlation
    try:
        corr_display = corr.copy()
        corr_display = corr_display.apply(pd.to_numeric, errors='coerce')
        corr_display = corr_display.replace([np.inf, -np.inf], np.nan)

        if corr_display.isna().all().all():
            print("Warning: Correlation matrix contains no valid numeric data. Skipping heatmap.")
        else:
            np.fill_diagonal(corr_display.values, np.nan)
            corr_display = corr_display.astype(float)
            corr_plot = corr_display.fillna(0.0)

            colors = [
                (0.0, (0.0, 0.0, 0.0, 0.0)),
                (0.5, (0.32, 0.72, 0.53, 1.0)),
                (1.0, (0.0, 0.0, 0.0, 0.0))
            ]
            cmap = LinearSegmentedColormap.from_list('green_transparent', [c[1] for c in colors], N=256)

            fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
            ax.set_facecolor('#1a1a1a')

            mask_annot = np.eye(len(corr_plot), dtype=bool)
            annot_data = corr_display.copy()
            annot_data[mask_annot] = np.nan

            sns.heatmap(corr_plot, annot=annot_data, fmt='.2f', cmap=cmap,
                        center=0, square=True, linewidths=1, linecolor='#2a2a2a',
                        cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, ax=ax,
                        annot_kws={'color': 'white', 'fontsize': 10})

            plt.title(f'{portfolio_name} - Correlation Matrix', fontsize=16, fontweight='light', color='white', pad=20)
            ax.tick_params(colors='white')
            plt.setp(ax.get_xticklabels(), color='white', rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), color='white')

            plt.tight_layout()
            plt.savefig('out/out_visualise/correlation.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
            plt.close()
    except Exception as e:
        print(f"Error while plotting correlation heatmap: {e}")

    # Efficient Frontier
    optimal_return = stats_df[stats_df['portfolio_type'] == 'Optimal']['expected_return'].iloc[0]
    optimal_std = stats_df[stats_df['portfolio_type'] == 'Optimal']['volatility'].iloc[0]
    optimal_sharpe = stats_df[stats_df['portfolio_type'] == 'Optimal']['sharpe_ratio'].iloc[0]

    current_return = stats_df[stats_df['portfolio_type'] == 'Current']['expected_return'].iloc[0]
    current_std = stats_df[stats_df['portfolio_type'] == 'Current']['volatility'].iloc[0]
    current_sharpe = stats_df[stats_df['portfolio_type'] == 'Current']['sharpe_ratio'].iloc[0]

    colors_green = ['#90EE90', '#52B788', '#2D6A4F', '#1B4332']
    cmap_green = LinearSegmentedColormap.from_list('green_gradient', colors_green)

    sharpe_min = results_df['sharpe_ratio'].min()
    sharpe_max = results_df['sharpe_ratio'].max()

    optimal_color_value = (optimal_sharpe - sharpe_min) / (sharpe_max - sharpe_min)
    optimal_color = cmap_green(optimal_color_value)

    current_color_value = (current_sharpe - sharpe_min) / (sharpe_max - sharpe_min)
    current_red = '#8B0000' if current_color_value < 0.5 else '#FF6B6B'

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.weight'] = 'light'

    scatter = plt.scatter(results_df['volatility'], results_df['returns'],
                          c=results_df['sharpe_ratio'], cmap=cmap_green,
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
    plt.savefig('out/out_visualise/efficient_frontier.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print("Visualisation complete!")


if __name__ == "__main__":
    main()