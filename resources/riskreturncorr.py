import stockinfo as SI
import pandas as pd
from datetime import datetime
import os

assets = input("list of asset, comma separated: ").split(",")
duration = input("duration: ")
interval = input("interval: ")

# Initialize an empty DataFrame with 'Date' as the index
stockmatrix = pd.DataFrame()

# Loop through each asset and add its interval returns as a new column
for asset in assets:
    # Get the interval returns for the asset
    df = SI.StockData(asset).get_interval_returns(duration, interval)
    
    # Ensure the 'Date' column is correctly set as the index
    df = df.dropna().reset_index(drop=True)
    df.columns = ['Date', 'Returns']
    df.set_index('Date', inplace=True)
    df.rename(columns={'Returns': asset}, inplace=True)
    
    # Merge with the main DataFrame
    if stockmatrix.empty:
        stockmatrix = df
    else:
        stockmatrix = stockmatrix.join(df, how='outer')

# Reset the index to make 'Date' a column again
stockmatrix.reset_index(inplace=True)

# Calculate mean and standard deviation for each column
stats = {
    'Mean': stockmatrix.mean(),
    'Standard Deviation': stockmatrix.std()
}

# Create a new DataFrame for the statistics
stats_df = pd.DataFrame(stats)

# Generate the filename for the statistics
assets_str = ''.join(assets)
date_str = datetime.now().strftime("%d%m%Y%H%M")
stats_filename = f"{assets_str}_meanstdev_{date_str}.csv"

# Define the directory to save the file
directory = "/Users/xizo/python-project---financial-analytics/stats"

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Full path to save the statistics file
stats_filepath = os.path.join(directory, stats_filename)

# Save the statistics DataFrame to a CSV file
stats_df.to_csv(stats_filepath)

# Display the statistics DataFrame
print(stats_df)
print(f"Statistics saved to {stats_filepath}")

# Calculate the correlation matrix
correlation_matrix = stockmatrix.corr()

# Generate the filename for the correlation matrix
corrmat_filename = f"{assets_str}_corrmat_{date_str}.csv"

# Full path to save the correlation matrix file
corrmat_filepath = os.path.join(directory, corrmat_filename)

# Save the correlation matrix to a CSV file
correlation_matrix.to_csv(corrmat_filepath)

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)
print(f"Correlation matrix saved to {corrmat_filepath}")