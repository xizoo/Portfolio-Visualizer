import stockinfo as SI
import pandas as pd
from datetime import datetime
import os
import numpy as np

class Portfolio():
    def __init__(self, assets,duration,interval):
        self.assets = assets
        self.duration = duration
        self.interval = interval
        self.stockmatrix = pd.DataFrame()
        self.initialize_weights()
        self._fetch_data()

    def add_assets(self,asset):
        self.assets.append(asset)
        return f"{asset} added successfully"
    
    def remove_assets(self,asset):
        self.assets.remove(asset)
        return f"{asset} removed successfully"

    def _fetch_data(self):
        for asset in self.assets:
            df = SI.Asset(asset).get_interval_returns(self.duration, self.interval)
            df = df.dropna().reset_index(drop=True)
            df.columns = ['Date', 'Returns']
            df.set_index('Date', inplace=True)
            df.rename(columns={'Returns': asset}, inplace=True)
            if self.stockmatrix.empty:
                self.stockmatrix = df
            else:
                self.stockmatrix = self.stockmatrix.join(df, how='outer')
        self.stockmatrix.reset_index(inplace=True)

    def calculate_mean_std(self):
        stats = {
            'Mean': self.stockmatrix.mean(),
            'Standard Deviation': self.stockmatrix.std()
        }
        stats_df = pd.DataFrame(stats)
        return stats_df

    def calculate_correlation(self):
        correlation_matrix = self.stockmatrix.corr()
        return correlation_matrix

    def save_to_csv(self, df, filename_suffix):
        assets_str = ''.join(self.assets)
        date_str = datetime.now().strftime("%d%m%Y%H%M")
        filename = f"{assets_str}_{filename_suffix}_{date_str}.csv"
        directory = "/Users/xizo/python-project---financial-analytics/stats"
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")

    def initialize_weights(self):
        num_assets = len(self.assets)
        weight = 1 / num_assets
        self.weights = {asset: float(weight) for asset in self.assets}

    def modify_weights(self):
        for asset in self.assets:
            new_weight = input(f"Enter the weight for {asset}: ")
            self.weights[asset] = new_weight

    def mve_finder(self):
        pass
    
    def portfolio_statistics(self):
        risk_free_rate = 0.01  # Assuming a risk-free rate of 1%
        mean_returns = self.stockmatrix.mean()
        cov_matrix = self.stockmatrix.cov()
        weights = np.array(list(self.weights.values()))
        if weights.shape[0] != len(self.assets):
            raise ValueError("Number of weights does not match number of assets")

        portfolio_return = np.sum(mean_returns * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_stddev

        # self.portfolio_return = portfolio_return
        # self.portfolio_stdev = portfolio_stddev
        # self.portfolio_sharpe = portfolio_sharpe

        return portfolio_return,portfolio_stddev



if __name__ == "__main__":
    assets_input = input("list of assets, comma separated: ")
    assets = [asset.strip() for asset in assets_input.split(',')]
    duration = input("duration: ")
    interval = input("interval: ")

    stock_stats = Portfolio(assets, duration, interval)

    # Calculate and save mean and standard deviation
    mean_std_df = stock_stats.calculate_mean_std()
    stock_stats.save_to_csv(mean_std_df, "meanstdev")

    # Calculate and save correlation matrix
    corr_matrix_df = stock_stats.calculate_correlation()
    
    # Remove 'Date' row and column if present
    if 'Date' in corr_matrix_df.columns:
        corr_matrix_df = corr_matrix_df.drop(columns='Date', index='Date')
    
    print(stock_stats.portfolio_statistics())