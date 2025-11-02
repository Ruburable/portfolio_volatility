import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    if not os.path.exists('out'):
        print("Visualisation failed")
        return

    calc_in = pd.read_csv('out/calc_in.csv')
    vol = pd.read_csv('out/vol.csv')
    corr = pd.read_csv('out/corr.csv', index_col=0)

    vol['date'] = pd.to_datetime(vol['date'])

    portfolio_name = calc_in['portfolio_name'].iloc[0]
    window = calc_in['window'].iloc[0]

    # Plot volatility
    plt.figure(figsize=(14, 7))
    plt.plot(vol['date'], vol['volatility'], linewidth=2, color='#2E86AB')
    plt.fill_between(vol['date'], vol['volatility'], alpha=0.3, color='#2E86AB')

    plt.title(f'{portfolio_name} - Rolling {window}-Day Annualized Volatility', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    mean_vol = vol['volatility'].mean()
    max_vol = vol['volatility'].max()
    min_vol = vol['volatility'].min()
    current_vol = vol['volatility'].iloc[-1]

    stats_text = f'Current: {current_vol:.2%}\nMean: {mean_vol:.2%}\nMax: {max_vol:.2%}\nMin: {min_vol:.2%}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('out/volatility.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)

    plt.title(f'{portfolio_name} - Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('out/correlation.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualisation complete!")


if __name__ == "__main__":
    main()