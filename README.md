# Portfolio Volatility Analyzer

A tool for measuring and analyzing portfolio volatility to help assess investment risk.

## Overview

This project calculates portfolio volatility metrics to help investors understand the risk characteristics of their investment portfolios. Volatility serves as a key indicator of price fluctuations and potential risk exposure.

## Features

- Calculate historical volatility of individual assets
- Compute portfolio-level volatility using correlation matrices
- Support for multiple volatility measures (standard deviation, variance)
- Visualization of volatility trends over time
- Rolling window volatility analysis

## Getting Started

### Prerequisites

- Python 3.8+
- pandas
- numpy
- matplotlib (for visualizations)

## Methodology

The analyzer uses the following approach:

1. **Returns Calculation**: Computes daily/monthly returns from price data
2. **Covariance Matrix**: Calculates asset correlations and covariances
3. **Portfolio Variance**: Applies portfolio weights to compute overall variance
4. **Annualization**: Converts volatility to annual terms for standardization