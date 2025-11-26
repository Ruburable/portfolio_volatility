import json
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import os


def main():
    os.makedirs('out/out_calculate', exist_ok=True)
    os.makedirs('out/out_optimise', exist_ok=True)

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

    # Download market benchmark (S&P 500)
    try:
        market_prices = yf.download('^GSPC', start=start_date, end=end_date, progress=False, auto_adjust=False)[
            'Adj Close']
    except:
        print("Warning: Market data download unsuccessful!")
        market_prices = None

    returns = prices.pct_change().dropna()

    # Calculate portfolio returns (weighted)
    portfolio_returns = (returns * weights).sum(axis=1)

    # Store calculation details
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

    # Calculate rolling portfolio volatility
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

    # Create correlation matrix
    correlation_matrix = returns.corr()
    corr = correlation_matrix.copy()

    # Save calculation outputs
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

    # Calculate historical metrics
    historical_portfolio_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    historical_portfolio_vol = portfolio_returns.std() * np.sqrt(252)

    if market_prices is not None:
        market_returns = market_prices.pct_change().dropna()
        common_dates = portfolio_returns.index.intersection(market_returns.index)
        portfolio_returns_aligned = portfolio_returns.loc[common_dates]
        market_returns_aligned = market_returns.loc[common_dates]

        # Calculate alpha (portfolio excess return vs market)
        market_mean_return = market_returns_aligned.mean() * 252
        alpha = current_return - market_mean_return

        # Calculate beta
        covariance = np.cov(portfolio_returns_aligned, market_returns_aligned)[0, 1]
        market_variance = market_returns_aligned.var()
        beta = covariance / market_variance
    else:
        alpha = None
        beta = None

    # Calculate distance metrics between current and optimal
    return_distance = optimal_return - current_return
    volatility_distance = optimal_std - current_std
    sharpe_distance = optimal_sharpe - current_sharpe

    return_distance_pct = (return_distance / current_return) * 100
    volatility_distance_pct = (volatility_distance / current_std) * 100
    sharpe_distance_pct = (sharpe_distance / current_sharpe) * 100

    # Weights difference (L1 norm)
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

    # Distance metrics for visualization
    distance_metrics = pd.DataFrame({
        'metric': [
            'Expected Return Distance',
            'Volatility Distance',
            'Sharpe Ratio Distance',
            'Return Distance (%)',
            'Volatility Distance (%)',
            'Sharpe Distance (%)',
            'Total Weights Change (L1)',
            'Alpha',
            'Beta'
        ],
        'value': [
            return_distance,
            volatility_distance,
            sharpe_distance,
            return_distance_pct,
            volatility_distance_pct,
            sharpe_distance_pct,
            weights_distance,
            alpha if alpha is not None else np.nan,
            beta if beta is not None else np.nan
        ]
    })

    # Gradient bar data (normalized 0-1 for visualization)
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

    # Save optimization outputs
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

    return calc_in, vol, corr, optimal_df, stats_df


if __name__ == "__main__":
    calc_in, vol, corr, optimal_df, stats_df = main()