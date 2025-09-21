"""
portfolio/optimizer.py

Core portfolio optimization logic.
This module contains functions to run portfolio optimization
based on stock tickers and return expected returns, risk, and weights.
"""

import matplotlib
matplotlib.use("Agg")   # Non-GUI backend for server environments

import numpy as np
import matplotlib.pyplot as plt

def optimize_portfolio(stocks: list):
    """
    Optimize portfolio weights given a list of stock tickers.

    Args:
        stocks (list): List of stock tickers. Example: ["AAPL", "TSLA", "MSFT"]

    Returns:
        results (dict): Expected return, volatility, and optimized weights.
        plots (dict): Matplotlib figures (not yet serialized).
    """

    n = len(stocks)
    if n == 0:
        raise ValueError("Stock list cannot be empty")

    # --- Placeholder logic: equal weights ---
    weights = {ticker: 1/n for ticker in stocks}

    # Example: random return & volatility for MVP
    expected_return = round(np.random.uniform(0.05, 0.15), 4)
    volatility = round(np.random.uniform(0.05, 0.20), 4)

    results = {
        "expected_return": expected_return,
        "volatility": volatility,
        "weights": weights
    }

    # --- Generate plots ---
    plots = {}

    # 1. Portfolio Allocation Pie Chart
    fig1, ax1 = plt.subplots()
    ax1.pie(weights.values(), labels=weights.keys(), autopct="%1.1f%%", startangle=90)
    ax1.set_title("Portfolio Allocation")
    plots["pie_chart"] = fig1

    # 2. Efficient Frontier Scatter (mocked for now)
    # Simulate 100 random portfolios
    returns = np.random.uniform(0.05, 0.20, 100)
    risks = np.random.uniform(0.05, 0.25, 100)

    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(risks, returns, c=returns/risks, cmap="viridis", alpha=0.7)
    ax2.scatter(volatility, expected_return, c="red", marker="*", s=200, label="Optimized Portfolio")
    ax2.set_xlabel("Volatility (Risk)")
    ax2.set_ylabel("Expected Return")
    ax2.set_title("Efficient Frontier (Simulated)")
    ax2.legend()
    plots["efficient_frontier"] = fig2

    return results, plots
