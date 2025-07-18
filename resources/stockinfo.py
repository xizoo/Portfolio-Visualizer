import yfinance as yf
import pandas as pd
from datetime import datetime
import os

class StockData:
    """
    A class to fetch and analyze stock data using Yahoo Finance API.

    Attributes:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        print(f"Initialized StockData class for ticker: {self.ticker}")

    # Function to get_interval_returns
    def get_interval_returns(self, duration: str, interval: str) -> pd.DataFrame:
        print(f"\nFetching interval returns for {self.ticker}...")
        # Obtain stock info for given duration and interval
        stock = yf.download(tickers=self.ticker, period=duration, interval=interval)
        # Catch error if no stock
        if stock.empty:
            print("No data found. Check inputs.")
            return None
        # Calculate Returns of Adj Close Price
        stock['Returns'] = stock['Close'].pct_change()
        # Remove NA Values, only keep returns
        returns_df = stock[['Returns']].dropna().reset_index()
        # Stopped saving CSV to save storage
        #self.save_to_csv(returns_df, "intervalreturns", duration, interval)
        return returns_df
    
    def get_total_return(self, duration: str, interval: str) -> pd.DataFrame:
        """
        Computes total return between each interval.
        Adjusts dividends time format to align with price data.

        Formula:
        Total Return (%) = [(Next Close + Dividend Yield (if any)) - Previous Close] / Previous Close * 100

        Parameters:
            duration (str): The duration of data to fetch (e.g., '5y', 'max').
            interval (str): The interval of data (e.g., '1d', '1mo').

        Returns:
            pd.DataFrame: A DataFrame with the total return percentages.
        """
        print(f"\nFetching total returns for {self.ticker}...")

        # Fetch price and dividend data
        print(self.get_price(duration, interval, pricetype="close"))
        stock_prices = self.get_price(duration, interval, pricetype="close").reset_index()  # Reset index
        dividends = self.get_dividend(duration).reset_index()  # Reset index

        # Convert dividend 'Date' column to match stock_prices format (YYYY-MM-DD)
        dividends['Date'] = dividends['Date'].apply(lambda x: str(x).split(" ")[0])  # Keep only the date portion

        # Flatten multi-level columns in stock_prices
        stock_prices.columns = stock_prices.columns.droplevel(1)  # Keep only the second level of the header

        def calculate_adjusted_price(price_df, dividend_df, interval: str) -> pd.DataFrame:
            interval_map = {
                '1d': 'D',
                '1wk': 'W',
                '1mo': 'M',
            }
            resample_freq = interval_map.get(interval.lower(), 'M')  # Default to 'M' if interval is not in the map

            """
            Adjusts the price DataFrame to account for dividends, aggregated by a given interval.

            Parameters:
                price_df (pd.DataFrame): DataFrame containing stock prices with columns ['Date', 'Close'].
                dividend_df (pd.DataFrame): DataFrame containing dividends with columns ['Date', 'Dividend'].
                interval (str): Interval for resampling dividends (e.g., '1M' for monthly, '1D' for daily).

            Returns:
                pd.DataFrame: Updated price DataFrame with an additional column ['Adj Price'].
            """
            # Ensure both Date columns are consistent in format
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            dividend_df['Date'] = pd.to_datetime(dividend_df['Date'])

            # Set Date as the index for dividend DataFrame
            dividend_df = dividend_df.set_index('Date')

            # Resample dividends to the specified interval and sum within each interval
            resampled_dividends = dividend_df.resample(resample_freq).sum().reset_index()

            # Merge the resampled dividend data with the price data
            merged_df = price_df.merge(resampled_dividends, on='Date', how='left')

            # Fill NaN values in the 'Dividend' column with 0
            merged_df['Dividend'] = merged_df['Dividend'].fillna(0)

            # Create the 'Adj Price' column: Close + Dividend
            merged_df['Adj Price'] = merged_df['Close'] + merged_df['Dividend']

            return merged_df

        result=calculate_adjusted_price(stock_prices, dividends,interval)

        # Save to CSV
        self.save_to_csv(result, "returns", duration, interval)
        return result


    def save_to_csv(self, df: pd.DataFrame, datatype: str, duration: str, interval: str):
        """
        Save a DataFrame to a CSV file with a standardized naming convention.

        Parameters:
            df (pd.DataFrame): The DataFrame to save.
            datatype (str): Type of data (e.g., 'allprices', 'intervalreturns', 'balancesheet').
            duration (str): Duration for the data (e.g., '5y', 'max') or empty if not applicable.
            interval (str): Interval for the data (e.g., '1mo') or empty if not applicable.
        """
        if df is not None:
            # Resolve project root and output directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            output_folder = os.path.join(project_root, "data")
            os.makedirs(output_folder, exist_ok=True)

            # Handle invalid or irrelevant duration/interval
            duration_label = duration if duration and duration != "N/A" else ""
            interval_label = interval if interval and interval != "quarterly" else ""

            # Generate the file name
            timestamp = datetime.now().strftime("%d%m%Y%H%M")
            file_name = f"{self.ticker}_{datatype}_{timestamp}_{duration_label}_{interval_label}.csv".strip("_")

            # Save file to /data folder
            file_path = os.path.join(output_folder, file_name)
            df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")



aapl=StockData("AAPL")
return_test= aapl.get_interval_returns('1y','1wk')