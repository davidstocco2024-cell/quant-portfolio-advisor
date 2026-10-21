import yfinance as yf
import pandas as pd

def fetch_market_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads adjusted closing prices for a list of tickers.
    Adjusted prices are crucial for accurate financial return calculations.
    """
    print(f"Fetching data for: {tickers} from {start_date} to {end_date}...")
    
    # auto_adjust=True is mandatory to account for dividends and stock splits
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    # If a single ticker is passed, yfinance returns a Series. We convert it to a DataFrame.
    if isinstance(data, pd.Series):
        data = data.to_frame()
        
    # Drop days where no asset traded (e.g., weekends/holidays)
    data = data.dropna(how='all')
    
    # Forward-fill missing data (e.g., if one asset didn't trade on a specific day)
    data = data.ffill()
    
    return data