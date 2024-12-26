from resources.stockinfo import StockData
#addcomment
def display_duration_options():
    """Display all available duration options for the user."""
    print("\nAvailable Duration Options:")
    print(" - 1d   : Last 1 day")
    print(" - 5d   : Last 5 days")
    print(" - 1mo  : Last 1 month")
    print(" - 3mo  : Last 3 months")
    print(" - 6mo  : Last 6 months")
    print(" - 1y   : Last 1 year")
    print(" - 2y   : Last 2 years")
    print(" - 5y   : Last 5 years")
    print(" - 10y  : Last 10 years")
    print(" - 30y  : Last 30 years")
    print(" - max  : All available data\n")
    

def get_valid_duration():
    while True:
        duration = input("Enter the duration (or type 'options' to display all options): ").strip().lower()
        if duration == "options":
            display_duration_options()
        else:
            return duration

def main():
    print("\n--- Welcome to the Stock Data Program ---\n")
    ticker = input("Enter the stock ticker symbol (e.g., AAPL): ").strip().upper()

    stock = StockData(ticker)

    print("\nSelect the type of program to run:")
    print("1. Interval Returns")
    print("2. Price History")
    print("3. Balance Sheet")
    print("4. Divident")
    print("5. All (Run all 4 programs: Interval Returns, Price History, Balance Sheet)")
    choice = input("Enter your choice: ").strip()

    duration = interval = pricetype = None
    if choice in ["1", "2", "5"]:
        duration = get_valid_duration()
        interval = input("Enter the data interval (e.g., 1d, 1wk, 1mo): ").strip()

    if choice == "1":
        print("\n--- Running Interval Returns Program ---")
        stock.get_interval_returns(duration, interval)
    elif choice == "2":
        print("\n--- Running Price History Program ---")
        pricetype = input("Enter the price type (e.g., Close, Open, High, Low) or press Enter for all: ").strip()
        stock.get_price(duration, interval, pricetype)
    elif choice == "3":
        print("\n--- Running Balance Sheet Program ---")
        stock.get_balance()
    elif choice == "4":
        print("\n--- Running Divident Program ---")
        duration = get_valid_duration()
        stock.get_dividend(duration)
    elif choice == "5":
        print("\n--- Running All Programs ---")
        stock.get_interval_returns(duration, interval)
        pricetype = input("\nEnter the price type (e.g., Close, Open, High, Low) or press Enter for all: ").strip()
        stock.get_price(duration, interval, pricetype)
        stock.get_balance()
        stock.get_dividend(duration)
    else:
        print("Invalid choice. Please restart the program.")

if __name__ == "__main__":
    main()
