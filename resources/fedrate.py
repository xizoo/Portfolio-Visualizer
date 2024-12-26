from pandas_datareader import data as pdr
import datetime

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2023, 12, 31)

# Fetch Fed Funds Rate from FRED
fed_rate = pdr.DataReader('FEDFUNDS', 'fred', start, end)
print(fed_rate.head())
