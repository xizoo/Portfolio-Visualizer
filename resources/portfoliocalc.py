import pandas as pd
import stockinfo as SI
import numpy as np

class Asset:
    def __init__(self, rtn, risk):
        self.rtn = rtn
        self.risk = risk
        self.corr = {}

    def set_corr(self, asset, corr):
        self.corr[asset] = corr
        asset.corr[self] = corr

    def get_corr(self, asset):
        return self.corr[asset]

class Portfolio:
    def __init__(self, assets, weights):
        self.assets = assets
        self.weights = np.array(weights)
        self.stdevs = np.array([asset.risk for asset in assets])
        self.corr_matrix = self._build_corr_matrix()

    def _build_corr_matrix(self):
        size = len(self.assets)
        corr_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                corr_matrix[i, j] = self.assets[i].get_corr(self.assets[j])
                corr_matrix[j, i] = corr_matrix[i, j]
        return corr_matrix
    
    def weighted_return(self):
        return np.dot(self.weights, [asset.rtn for asset in self.assets])

    def weighted_risk(self):
        # Calculate the covariance matrix
        cov_matrix = np.outer(self.stdevs, self.stdevs) * self.corr_matrix
        
        # Calculate the portfolio variance
        portfolio_variance = np.dot(self.weights.T, np.dot(cov_matrix, self.weights))
        
        # Calculate the portfolio standard deviation (risk)
        portfolio_risk = np.sqrt(portfolio_variance)
        
        return portfolio_risk

# Example usage
aapl = Asset(0.12, 0.15)
msft = Asset(0.10, 0.10)
googl = Asset(0.14, 0.20)

aapl.set_corr(msft, 0.5)
aapl.set_corr(googl, 0.6)
msft.set_corr(googl, 0.7)

assets = [aapl, msft, googl]
weights = [0.4, 0.3, 0.3]

portfolio = Portfolio(assets, weights)
print("Portfolio Return:", portfolio.weighted_return())
print("Portfolio Risk:", portfolio.weighted_risk())
