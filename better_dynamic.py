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
    returns_comparison = pd.read_csv('out/out_calculate/returns_comparison.csv')

    current_volatility = stats_df[stats_df['portfolio_type'] == 'Current']['volatility'].iloc[0]
    current_return = stats_df[stats_df['portfolio_type'] == 'Current']['expected_return'].iloc[0]
    current_sharpe = stats_df[stats_df['portfolio_type'] == 'Current']['sharpe_ratio'].iloc[0]

    # Filter portfolios with same volatility (±2% tolerance)
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

    # Calculate improvement metrics
    sharpe_improvement = ((better_sharpe - current_sharpe) / current_sharpe) * 100
    return_improvement = ((better_return - current_return) / current_return) * 100
    volatility_diff = ((better_volatility - current_volatility) / current_volatility) * 100

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'portfolio_type': ['Current', 'Better Dynamic'],
        'expected_return': [current_return, better_return],
        'volatility': [current_volatility, better_volatility],
        'sharpe_ratio': [current_sharpe, better_sharpe]
    })

    improvements_df = pd.DataFrame({
        'metric': ['Sharpe Ratio Improvement (%)', 'Return Improvement (%)', 'Volatility Difference (%)'],
        'value': [sharpe_improvement, return_improvement, volatility_diff]
    })

    # Historical performance comparison
    portfolio_cumulative = returns_comparison['portfolio_cumulative'].iloc[-1] - 1
    market_cumulative = returns_comparison['market_cumulative'].iloc[-1] - 1

    historical_comparison = pd.DataFrame({
        'portfolio_type': ['Current Portfolio', 'Market (S&P 500)'],
        'historical_total_return': [portfolio_cumulative, market_cumulative],
        'historical_volatility': [
            returns_comparison['portfolio_returns'].std() * np.sqrt(252),
            returns_comparison['market_returns'].std() * np.sqrt(252)
        ]
    })

    # Summary statistics
    summary_stats = pd.DataFrame({
        'metric': [
            'Current Sharpe Ratio',
            'Better Dynamic Sharpe Ratio',
            'Target Volatility',
            'Better Portfolio Volatility',
            'Expected Return Increase',
            'Historical Portfolio Return',
            'Historical Market Return'
        ],
        'value': [
            f'{current_sharpe:.4f}',
            f'{better_sharpe:.4f}',
            f'{current_volatility:.2%}',
            f'{better_volatility:.2%}',
            f'{return_improvement:.2f}%',
            f'{portfolio_cumulative:.2%}',
            f'{market_cumulative:.2%}'
        ]
    })

    # Save outputs
    comparison_df.to_csv('out/out_dynamic/portfolio_comparison.csv', index=False)
    improvements_df.to_csv('out/out_dynamic/improvements.csv', index=False)
    historical_comparison.to_csv('out/out_dynamic/historical_comparison.csv', index=False)
    summary_stats.to_csv('out/out_dynamic/summary_stats.csv', index=False)

    # Save the index of the better portfolio for potential weight reconstruction
    better_portfolio_info = pd.DataFrame({
        'simulation_index': [best_sharpe_idx],
        'expected_return': [better_return],
        'volatility': [better_volatility],
        'sharpe_ratio': [better_sharpe],
        'sharpe_improvement_pct': [sharpe_improvement],
        'return_improvement_pct': [return_improvement]
    })
    better_portfolio_info.to_csv('out/out_dynamic/better_portfolio_info.csv', index=False)

    print("Better dynamic portfolio analysis complete!")
    print(f"Sharpe Ratio Improvement: {sharpe_improvement:.2f}%")
    print(f"Expected Return Improvement: {return_improvement:.2f}%")
    print(f"Volatility Difference: {volatility_diff:.2f}%")

    return comparison_df, improvements_df


if __name__ == "__main__":
    comparison, improvements = main()