import stockinfo as SI
import pandas as pd
from datetime import datetime
import os

class StockStats:
    def __init__(self, assets, duration, interval):
        self.assets = assets
        self.duration = duration
        self.interval = interval
        self.stockmatrix = pd.DataFrame()
        self._fetch_data()

    def _fetch_data(self):
        for asset in self.assets:
            df = SI.StockData(asset).get_interval_returns(self.duration, self.interval)
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

if __name__ == "__main__":
    assets = input("list of asset, comma separated: ").split(",")
    duration = input("duration: ")
    interval = input("interval: ")

    stock_stats = StockStats(assets, duration, interval)

    # Calculate and save mean and standard deviation
    mean_std_df = stock_stats.calculate_mean_std()
    stock_stats.save_to_csv(mean_std_df, "meanstdev")

    # Calculate and save correlation matrix
    corr_matrix_df = stock_stats.calculate_correlation()
    
    # Remove 'Date' row and column if present
    if 'Date' in corr_matrix_df.columns:
        corr_matrix_df = corr_matrix_df.drop(columns='Date', index='Date')
    
    stock_stats.save_to_csv(corr_matrix_df, "corrmat")