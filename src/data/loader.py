"""
Data Loading Module
Handles fetching historical data for both Equities and Crypto.
"""
import yfinance as yf # type: ignore
import pandas as pd
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta

class MarketDataLoader:
    """
    Unified data loader for hybrid assets.
    """
    
    def __init__(self, lookback_days: int = 365):
        self.lookback_days = lookback_days
        self.start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    def fetch_data(self, tickers: List[str], asset_class: str) -> pd.DataFrame:
        """
        Fetches OHLCV data.
        
        Args:
            tickers: List of symbols (e.g., ['AAPL', 'MSFT'] or ['BTC-USD'])
            asset_class: 'equity' or 'crypto'
            
        Returns:
            MultiIndex DataFrame compatible with vectorbt.
        """
        print(f"Fetching {asset_class} data for: {tickers}")
        
        try:
            # yfinance handles both, but we need to be careful with suffixes for crypto
            data = yf.download(
                tickers, 
                start=self.start_date, 
                group_by='ticker', 
                auto_adjust=True,
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {tickers}")

            # Data Cleaning & Validation
            data = self._validate_structure(data, tickers)
            
            return data

        except Exception as e:
            print(f"Error fetching data: {e}")
            raise

    def _validate_structure(self, df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Ensures DataFrame has correct MultiIndex structure."""
        if len(tickers) == 1:
            # Reformat single ticker to match multi-ticker structure
            ticker = tickers[0]
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        
        # Drop rows with all NaNs
        df.dropna(how='all', inplace=True)
        return df

if __name__ == "__main__":
    # Smoke Test
    loader = MarketDataLoader(lookback_days=30)
    
    # Test Equities
    stocks = loader.fetch_data(['AAPL', 'MSFT'], 'equity')
    print(f"Stocks Shape: {stocks.shape}")
    
    # Test Crypto
    crypto = loader.fetch_data(['BTC-USD', 'ETH-USD'], 'crypto')
    print(f"Crypto Shape: {crypto.shape}")
