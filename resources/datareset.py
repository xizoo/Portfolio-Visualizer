import os
import glob

# Define the directory containing the CSV files
data_directory = "/Users/xizo/python-project---financial-analytics/data"

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(data_directory, "*.csv"))

# Delete each CSV file
for file in csv_files:
    os.remove(file)
    print(f"Deleted {file}")

print("All CSV files in the data folder have been deleted.")