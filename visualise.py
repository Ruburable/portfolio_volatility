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

    # Plot volatility with improved styling
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 7))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.weight'] = 'light'

    plt.plot(vol['date'], vol['volatility'], linewidth=2, color='#52B788')
    plt.fill_between(vol['date'], vol['volatility'], alpha=0.4, color='#52B788')

    plt.title(f'Rolling {window}-Day Annualized Volatility', fontsize=14, fontweight='light', color='white')
    plt.xlabel('Date', fontsize=11, fontweight='light', color='white')
    plt.ylabel('Annualized Volatility', fontsize=11, fontweight='light', color='white')
    plt.grid(True, alpha=0.2, linestyle='--', color='gray')

    # Set y-axis to start at 0
    plt.ylim(bottom=0)

    # Set x-axis limits to data range (no gaps)
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
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    # Create a mask for diagonal values
    corr_display = corr.copy()
    np.fill_diagonal(corr_display.values, np.nan)

    # Create custom colormap: transparent at edges (-1, 1), green at center (0)
    # Define colors at key positions
    colors = [
        (0.0, (0.0, 0.0, 0.0, 0.0)),  # -1: fully transparent
        (0.5, (0.32, 0.72, 0.53, 1.0)),  # 0: solid green (#52B788)
        (1.0, (0.0, 0.0, 0.0, 0.0))  # 1: fully transparent
    ]

    cmap = LinearSegmentedColormap.from_list('green_transparent',
                                             [c[1] for c in colors],
                                             N=256)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    # Create mask for annotations (hide diagonal)
    mask_annot = np.eye(len(corr), dtype=bool)
    annot_data = corr.copy()
    annot_data[mask_annot] = np.nan

    sns.heatmap(corr_display, annot=annot_data, fmt='.2f', cmap=cmap,
                center=0, square=True, linewidths=1, linecolor='#2a2a2a',
                cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, ax=ax,
                annot_kws={'color': 'white', 'fontsize': 10})

    plt.title(f'{portfolio_name} - Correlation Matrix', fontsize=16, fontweight='light', color='white', pad=20)
    ax.tick_params(colors='white')
    plt.setp(ax.get_xticklabels(), color='white')
    plt.setp(ax.get_yticklabels(), color='white')

    plt.tight_layout()
    plt.savefig('out/out_visualise/correlation.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print("Visualisation complete!")


if __name__ == "__main__":
    main()