from pandas_datareader import data as pdr
import requests
from datetime import datetime as dt

#Different rates that you can extract, or use your own
treasury_1 = "DGS1"
treasury_3 = "DGS3"
treasury_5 = "DGS5"
fed_rate = "EFFR"

def fetch_rate(id):
    # Define the API key and endpoint
    api_key = '52c0724c62e67f6e8fae558c6a37fe5d'
    series_id = id
    start_date = '2000-01-01'
    end_date = dt.today().strftime('%Y-%m-%d')
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}'

    # Fetch data from FRED API
    response = requests.get(url)
    data = response.json()

    # Check if 'observations' key is in the response
    if 'observations' in data:
        # Extract the latest fund rate for the 1-year treasury
        observations = data['observations']
        latest_observation = observations[-1]
        latest_date = latest_observation['date']
        latest_rate = latest_observation['value']
        print(f"Latest Rate on {end_date}: {latest_rate}")
        return latest_rate
    else:
        print("Error: 'observations' key not found in the response")
        print(data)

fetch_rate(fed_rate)