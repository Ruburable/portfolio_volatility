import json
import pandas as pd
import numpy as np
import os


def main():
    os.makedirs('out/out_dynamic', exist_ok=True)

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio = data['portfolio']

    tickers = portfolio['tickers']
    current_weights = np.array(portfolio['weights'])

    stats_df = pd.read_csv('out/out_optimise/portfolio_stats.csv')
    results_df = pd.read_csv('out/out_optimise/simulation_results.csv')

    current_volatility = stats_df[stats_df['portfolio_type'] == 'Current']['volatility'].iloc[0]
    current_return = stats_df[stats_df['portfolio_type'] == 'Current']['expected_return'].iloc[0]
    current_sharpe = stats_df[stats_df['portfolio_type'] == 'Current']['sharpe_ratio'].iloc[0]

    optimal_volatility = stats_df[stats_df['portfolio_type'] == 'Optimal']['volatility'].iloc[0]
    optimal_return = stats_df[stats_df['portfolio_type'] == 'Optimal']['expected_return'].iloc[0]
    optimal_sharpe = stats_df[stats_df['portfolio_type'] == 'Optimal']['sharpe_ratio'].iloc[0]

    # Filter portfolios with same volatility as current (±2% tolerance)
    tolerance = 0.02
    similar_vol_mask = (results_df['volatility'] >= current_volatility * (1 - tolerance)) & \
                       (results_df['volatility'] <= current_volatility * (1 + tolerance))

    similar_vol_portfolios = results_df[similar_vol_mask].copy()

    if len(similar_vol_portfolios) == 0:
        print("No portfolios found with similar volatility!")
        return None

    # Find portfolio with best Sharpe ratio among similar volatility
    best_sharpe_idx = similar_vol_portfolios['sharpe_ratio'].idxmax()
    better_portfolio = similar_vol_portfolios.loc[best_sharpe_idx]

    better_return = better_portfolio['returns']
    better_volatility = better_portfolio['volatility']
    better_sharpe = better_portfolio['sharpe_ratio']

    # Calculate improvement metrics vs current
    sharpe_improvement = ((better_sharpe - current_sharpe) / current_sharpe) * 100
    return_improvement = ((better_return - current_return) / current_return) * 100
    volatility_diff = ((better_volatility - current_volatility) / current_volatility) * 100

    # Calculate metrics vs optimal
    sharpe_vs_optimal = ((better_sharpe - optimal_sharpe) / optimal_sharpe) * 100
    return_vs_optimal = ((better_return - optimal_return) / optimal_return) * 100
    volatility_vs_optimal = ((better_volatility - optimal_volatility) / optimal_volatility) * 100

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'portfolio_type': ['Current', 'Better Dynamic', 'Optimal'],
        'expected_return': [current_return, better_return, optimal_return],
        'volatility': [current_volatility, better_volatility, optimal_volatility],
        'sharpe_ratio': [current_sharpe, better_sharpe, optimal_sharpe]
    })

    improvements_df = pd.DataFrame({
        'metric': [
            'Sharpe Ratio Improvement vs Current (%)',
            'Return Improvement vs Current (%)',
            'Volatility Difference vs Current (%)',
            'Sharpe Ratio vs Optimal (%)',
            'Return vs Optimal (%)',
            'Volatility vs Optimal (%)'
        ],
        'value': [
            sharpe_improvement,
            return_improvement,
            volatility_diff,
            sharpe_vs_optimal,
            return_vs_optimal,
            volatility_vs_optimal
        ]
    })

    # Summary statistics
    summary_stats = pd.DataFrame({
        'metric': [
            'Current Sharpe Ratio',
            'Better Dynamic Sharpe Ratio',
            'Optimal Sharpe Ratio',
            'Target Volatility',
            'Better Portfolio Volatility',
            'Optimal Portfolio Volatility',
            'Expected Return Increase vs Current',
            'Portfolios Analyzed',
            'Similar Volatility Portfolios Found'
        ],
        'value': [
            f'{current_sharpe:.4f}',
            f'{better_sharpe:.4f}',
            f'{optimal_sharpe:.4f}',
            f'{current_volatility:.2%}',
            f'{better_volatility:.2%}',
            f'{optimal_volatility:.2%}',
            f'{return_improvement:.2f}%',
            f'{len(results_df):,}',
            f'{len(similar_vol_portfolios):,}'
        ]
    })

    # Save outputs
    comparison_df.to_csv('out/out_dynamic/portfolio_comparison.csv', index=False)
    improvements_df.to_csv('out/out_dynamic/improvements.csv', index=False)
    summary_stats.to_csv('out/out_dynamic/summary_stats.csv', index=False)

    # Save the index of the better portfolio for potential weight reconstruction
    better_portfolio_info = pd.DataFrame({
        'simulation_index': [best_sharpe_idx],
        'expected_return': [better_return],
        'volatility': [better_volatility],
        'sharpe_ratio': [better_sharpe],
        'sharpe_improvement_pct': [sharpe_improvement],
        'return_improvement_pct': [return_improvement],
        'similar_volatility_count': [len(similar_vol_portfolios)]
    })
    better_portfolio_info.to_csv('out/out_dynamic/better_portfolio_info.csv', index=False)

    print("Better dynamic portfolio analysis complete!")
    print(f"Sharpe Ratio Improvement: {sharpe_improvement:.2f}%")
    print(f"Expected Return Improvement: {return_improvement:.2f}%")
    print(f"Volatility Difference: {volatility_diff:.2f}%")
    print(f"Found {len(similar_vol_portfolios):,} portfolios with similar volatility")

    return comparison_df, improvements_df


if __name__ == "__main__":
    comparison, improvements = main()