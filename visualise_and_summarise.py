import os
import base64
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def create_visualizations():
    if not os.path.exists('out/out_calculate'):
        print("Error: 'out/out_calculate' folder not found.")
        return False

    if not os.path.exists('out/out_optimise'):
        print("Error: 'out/out_optimise' folder not found.")
        return False

    os.makedirs('out/out_visualise', exist_ok=True)

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio_name = data['portfolio']['name']

    calc_in = pd.read_csv('out/out_calculate/calc_in.csv')
    vol = pd.read_csv('out/out_calculate/vol.csv')
    corr = pd.read_csv('out/out_calculate/corr.csv', index_col=0)

    stats_df = pd.read_csv('out/out_optimise/portfolio_stats.csv')
    results_df = pd.read_csv('out/out_optimise/simulation_results.csv')
    gradient_data = pd.read_csv('out/out_optimise/gradient_data.csv')
    distance_metrics = pd.read_csv('out/out_optimise/distance_metrics.csv')

    vol['date'] = pd.to_datetime(vol['date'])
    window = calc_in['window'].iloc[0]

    # Volatility
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
    corr_display = corr.copy()
    np.fill_diagonal(corr_display.values, np.nan)

    colors = [
        (0.0, (0.0, 0.0, 0.0, 0.0)),
        (0.5, (0.32, 0.72, 0.53, 1.0)),
        (1.0, (0.0, 0.0, 0.0, 0.0))
    ]

    cmap = LinearSegmentedColormap.from_list('green_transparent', [c[1] for c in colors], N=256)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

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

    # Efficient Frontier
    optimal_return = stats_df[stats_df['portfolio_type'] == 'Optimal']['expected_return'].iloc[0]
    optimal_std = stats_df[stats_df['portfolio_type'] == 'Optimal']['volatility'].iloc[0]
    optimal_sharpe = stats_df[stats_df['portfolio_type'] == 'Optimal']['sharpe_ratio'].iloc[0]

    current_return = stats_df[stats_df['portfolio_type'] == 'Current']['expected_return'].iloc[0]
    current_std = stats_df[stats_df['portfolio_type'] == 'Current']['volatility'].iloc[0]
    current_sharpe = stats_df[stats_df['portfolio_type'] == 'Current']['sharpe_ratio'].iloc[0]

    better_return = None
    better_std = None
    better_sharpe = None
    if os.path.exists('out/out_dynamic/portfolio_comparison.csv'):
        dynamic_df = pd.read_csv('out/out_dynamic/portfolio_comparison.csv')
        better_row = dynamic_df[dynamic_df['portfolio_type'] == 'Better Dynamic']
        if not better_row.empty:
            better_return = better_row['expected_return'].iloc[0]
            better_std = better_row['volatility'].iloc[0]
            better_sharpe = better_row['sharpe_ratio'].iloc[0]

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

    if better_return is not None:
        better_color_value = (better_sharpe - sharpe_min) / (sharpe_max - sharpe_min)
        better_color = cmap_green(better_color_value)
        plt.scatter(better_std, better_return, marker='P', color=better_color, s=250,
                    edgecolors='white', linewidth=2, label='Better Dynamic Portfolio', zorder=5)

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

    # Gradient Bars Comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor='#1a1a1a')

    # Sharpe Ratio Comparison
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a1a')
    current_sharpe_norm = gradient_data[gradient_data['metric'] == 'Current Sharpe']['normalized_value'].iloc[0]
    optimal_sharpe_norm = gradient_data[gradient_data['metric'] == 'Optimal Sharpe']['normalized_value'].iloc[0]

    ax1.barh(['Portfolio'], [1], color='#2a2a2a', height=0.6)
    ax1.barh(['Portfolio'], [optimal_sharpe_norm], color='#52B788', height=0.6, alpha=0.7)
    ax1.scatter([current_sharpe_norm], ['Portfolio'], color='#FF6B6B', s=300, zorder=5, marker='|', linewidths=4)
    ax1.scatter([optimal_sharpe_norm], ['Portfolio'], color='#52B788', s=300, zorder=5, marker='|', linewidths=4)

    ax1.set_xlim(0, 1)
    ax1.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='light', color='white', pad=10)
    ax1.set_xlabel('Performance (Normalized)', fontsize=10, fontweight='light', color='white')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add text labels
    ax1.text(current_sharpe_norm, 0, f'Current\n{current_sharpe:.3f}',
             ha='center', va='bottom', color='#FF6B6B', fontsize=9, fontweight='bold')
    ax1.text(optimal_sharpe_norm, 0, f'Optimal\n{optimal_sharpe:.3f}',
             ha='center', va='top', color='#52B788', fontsize=9, fontweight='bold')

    # Volatility Comparison
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a1a')
    current_vol_norm = gradient_data[gradient_data['metric'] == 'Current Volatility']['normalized_value'].iloc[0]
    optimal_vol_norm = gradient_data[gradient_data['metric'] == 'Optimal Volatility']['normalized_value'].iloc[0]

    ax2.barh(['Portfolio'], [1], color='#2a2a2a', height=0.6)
    ax2.barh(['Portfolio'], [optimal_vol_norm], color='#90EE90', height=0.6, alpha=0.7)
    ax2.scatter([current_vol_norm], ['Portfolio'], color='#FF6B6B', s=300, zorder=5, marker='|', linewidths=4)
    ax2.scatter([optimal_vol_norm], ['Portfolio'], color='#90EE90', s=300, zorder=5, marker='|', linewidths=4)

    ax2.set_xlim(0, 1)
    ax2.set_title('Volatility Comparison (Lower is Better)', fontsize=12, fontweight='light', color='white', pad=10)
    ax2.set_xlabel('Risk Level (Normalized)', fontsize=10, fontweight='light', color='white')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.text(current_vol_norm, 0, f'Current\n{current_std:.2%}',
             ha='center', va='bottom', color='#FF6B6B', fontsize=9, fontweight='bold')
    ax2.text(optimal_vol_norm, 0, f'Optimal\n{optimal_std:.2%}',
             ha='center', va='top', color='#90EE90', fontsize=9, fontweight='bold')

    # Alpha/Beta Display
    ax3 = axes[2]
    ax3.set_facecolor('#1a1a1a')
    ax3.axis('off')

    alpha_val = distance_metrics[distance_metrics['metric'] == 'Alpha']['value'].iloc[0]
    beta_val = distance_metrics[distance_metrics['metric'] == 'Beta']['value'].iloc[0]
    sharpe_dist = distance_metrics[distance_metrics['metric'] == 'Sharpe Distance (%)']['value'].iloc[0]

    metrics_text = f'Additional Metrics\n\n'
    metrics_text += f'Alpha: {alpha_val:.4f}\n'
    metrics_text += f'Beta: {beta_val:.4f}\n'
    metrics_text += f'Sharpe Improvement Potential: {sharpe_dist:.2f}%'

    ax3.text(0.5, 0.5, metrics_text, ha='center', va='center',
             color='white', fontsize=14, fontweight='light',
             bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8, edgecolor='#52B788', linewidth=2))

    plt.tight_layout()
    plt.savefig('out/out_visualise/gradient_comparison.jpg', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print("Visualisation complete!")
    return True


def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def create_html_report():
    if not os.path.exists('out/out_visualise'):
        print("Error: 'out/out_visualise' folder not found.")
        return False

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio_name = data['portfolio']['name']

    volatility_img = image_to_base64('out/out_visualise/volatility.jpg')
    correlation_img = image_to_base64('out/out_visualise/correlation.jpg')
    frontier_img = image_to_base64('out/out_visualise/efficient_frontier.jpg')
    gradient_img = image_to_base64('out/out_visualise/gradient_comparison.jpg')

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{portfolio_name} - Portfolio Analysis Deck</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0a0a0a;
                color: #ffffff;
                overflow: hidden;
                height: 100vh;
                display: flex;
                flex-direction: column;
            }}

            header {{
                text-align: center;
                padding: 30px 0 20px 0;
                border-bottom: 2px solid #52B788;
                background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            }}

            header h1 {{
                font-size: 2.5em;
                font-weight: 300;
                color: #52B788;
                margin-bottom: 5px;
            }}

            header .subtitle {{
                font-size: 1em;
                color: #aaaaaa;
                font-weight: 300;
            }}

            .dashboard {{
                flex: 1;
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 20px;
                padding: 20px;
                overflow: hidden;
            }}

            .chart-card {{
                background-color: #1a1a1a;
                border-radius: 15px;
                padding: 20px;
                display: flex;
                flex-direction: column;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                cursor: pointer;
                transition: transform 0.2s ease;
            }}

            .chart-card:hover {{
                transform: scale(1.02);
            }}

            .chart-title {{
                font-size: 1.3em;
                font-weight: 300;
                color: #52B788;
                margin-bottom: 15px;
                text-align: center;
            }}

            .chart-container {{
                flex: 1;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
            }}

            .chart-container img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                border-radius: 10px;
            }}

            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.95);
                justify-content: center;
                align-items: center;
            }}

            .modal.active {{
                display: flex;
            }}

            .modal-content {{
                max-width: 95%;
                max-height: 95%;
                object-fit: contain;
                border-radius: 10px;
            }}

            .close-modal {{
                position: absolute;
                top: 30px;
                right: 30px;
                font-size: 3em;
                color: #52B788;
                cursor: pointer;
                font-weight: 300;
                transition: transform 0.2s ease;
            }}

            .close-modal:hover {{
                transform: scale(1.2);
            }}

            @media (max-width: 1200px) {{
                .dashboard {{
                    grid-template-columns: 1fr;
                    grid-template-rows: repeat(4, 1fr);
                }}
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>{portfolio_name}</h1>
            <p class="subtitle">Portfolio Analysis Dashboard | Generated on {datetime.now().strftime('%B %d, %Y')}</p>
        </header>

        <div class="dashboard">
            <div class="chart-card" onclick="openModal('volatility')">
                <h3 class="chart-title">Portfolio Volatility Analysis</h3>
                <div class="chart-container">
                    <img id="volatility-img" src="data:image/jpeg;base64,{volatility_img}" alt="Volatility Chart">
                </div>
            </div>

            <div class="chart-card" onclick="openModal('correlation')">
                <h3 class="chart-title">Asset Correlation Matrix</h3>
                <div class="chart-container">
                    <img id="correlation-img" src="data:image/jpeg;base64,{correlation_img}" alt="Correlation Matrix">
                </div>
            </div>

            <div class="chart-card" onclick="openModal('frontier')">
                <h3 class="chart-title">Efficient Frontier & Optimization</h3>
                <div class="chart-container">
                    <img id="frontier-img" src="data:image/jpeg;base64,{frontier_img}" alt="Efficient Frontier">
                </div>
            </div>

            <div class="chart-card" onclick="openModal('gradient')">
                <h3 class="chart-title">Performance Metrics Comparison</h3>
                <div class="chart-container">
                    <img id="gradient-img" src="data:image/jpeg;base64,{gradient_img}" alt="Gradient Comparison">
                </div>
            </div>
        </div>

        <div class="modal" id="modal" onclick="closeModal()">
            <span class="close-modal">&times;</span>
            <img class="modal-content" id="modal-img">
        </div>

        <script>
            function openModal(imageId) {{
                const modal = document.getElementById('modal');
                const modalImg = document.getElementById('modal-img');
                const img = document.getElementById(imageId + '-img');

                modal.classList.add('active');
                modalImg.src = img.src;
            }}

            function closeModal() {{
                const modal = document.getElementById('modal');
                modal.classList.remove('active');
            }}

            document.addEventListener('keydown', (e) => {{
                if (e.key === 'Escape') closeModal();
            }});
        </script>
    </body>
    </html>
    """

    output_path = 'out/portfolio_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Report generated: {output_path}")
    return True


def main():
    if create_visualizations():
        create_html_report()


if __name__ == "__main__":
    main()