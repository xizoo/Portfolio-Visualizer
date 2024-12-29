import stockinfo as si
import pandas as pd
"""
def get_total_return(self, duration: str, interval: str) -> pd.DataFrame:
    
    Computes total return between each interval.
    Adjusts dividends time format to align with price data.

    Formula:
    Total Return (%) = [(Next Close + Dividend Yield (if any)) - Previous Close] / Previous Close * 100

    Parameters:
        duration (str): The duration of data to fetch (e.g., '5y', 'max').
        interval (str): The interval of data (e.g., '1d', '1mo').

    Returns:
        pd.DataFrame: A DataFrame with the total return percentages.
    
    print(f"\nFetching total returns for {self.ticker}...")

    # Fetch price and dividend data
    stock_prices = self.get_price(duration, interval, pricetype="close").reset_index()  # Reset index
    dividends = self.get_dividend(duration).reset_index()  # Reset index

    # Convert dividend 'Date' column to match stock_prices format (YYYY-MM-DD)
    dividends['Date'] = dividends['Date'].apply(lambda x: str(x).split(" ")[0])  # Keep only the date portion

    # Flatten multi-level columns in stock_prices
    stock_prices.columns = stock_prices.columns.droplevel(1)  # Keep only the second level of the header

    print(stock_prices.head())        
    print(dividends.head())
    

    result=calculate_adjusted_price(stock_prices, dividends,interval)

    # Save to CSV
    self.save_to_csv(result, "returns", duration, interval)
    return result

def calculate_adjusted_price(price_df, dividend_df, interval: str) -> pd.DataFrame:
    interval_map = {
        '1d': 'D',
        '1wk': 'W',
        '1mo': 'M',
    }
    resample_freq = interval_map.get(interval.lower(), 'M')  # Default to 'M' if interval is not in the map

    
    Adjusts the price DataFrame to account for dividends, aggregated by a given interval.

    Parameters:
        price_df (pd.DataFrame): DataFrame containing stock prices with columns ['Date', 'Close'].
        dividend_df (pd.DataFrame): DataFrame containing dividends with columns ['Date', 'Dividend'].
        interval (str): Interval for resampling dividends (e.g., '1M' for monthly, '1D' for daily).

    Returns:
        pd.DataFrame: Updated price DataFrame with an additional column ['Adj Price'].
    
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
    """