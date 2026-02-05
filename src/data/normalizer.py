"""
Data Normalization Module
Merges Crypto (24/7) and Stock (M-F) streams into a unified UTC daily index.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Literal

AssetClass = Literal["equity", "crypto"]


class HybridDataNormalizer:
    """
    Normalizes hybrid asset data into a unified UTC daily index.

    Stock data is forward-filled over weekends/holidays so the equity curve
    remains flat while Crypto trades 24/7.
    """

    def __init__(self) -> None:
        self._stock_data: Optional[pd.DataFrame] = None
        self._crypto_data: Optional[pd.DataFrame] = None

    def ingest(
        self,
        data: pd.DataFrame,
        asset_class: AssetClass
    ) -> None:
        """
        Ingest OHLCV data for a given asset class.

        Args:
            data: MultiIndex DataFrame with (ticker, OHLCV) columns.
            asset_class: 'equity' or 'crypto'.
        """
        # Ensure index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Normalize to UTC (remove timezone or convert)
        if data.index.tz is not None:
            data.index = data.index.tz_convert("UTC").tz_localize(None)

        # Normalize index to date only (no time component)
        data.index = pd.to_datetime(data.index.date)
        data.index.name = "date"

        if asset_class == "equity":
            self._stock_data = data.copy()
        elif asset_class == "crypto":
            self._crypto_data = data.copy()
        else:
            raise ValueError(f"Unknown asset_class: {asset_class}")

    def normalize(self) -> pd.DataFrame:
        """
        Create a unified DataFrame with daily UTC index.

        Logic:
        1. Build a complete daily date range from min to max date across both assets.
        2. Reindex Crypto data (should have all days).
        3. Reindex Stock data and forward-fill weekends/holidays.
        4. Merge into a single DataFrame.

        Returns:
            Unified DataFrame with all tickers, weekend gaps filled for equities.
        """
        if self._stock_data is None and self._crypto_data is None:
            raise ValueError("No data ingested. Call ingest() first.")

        # Determine the unified date range
        all_dates = self._get_unified_date_range()

        result_frames: list[pd.DataFrame] = []

        # Process stock data with forward-fill
        if self._stock_data is not None:
            stock_reindexed = self._stock_data.reindex(all_dates)
            stock_reindexed = stock_reindexed.ffill()
            result_frames.append(stock_reindexed)

        # Process crypto data (should already have weekend data)
        if self._crypto_data is not None:
            crypto_reindexed = self._crypto_data.reindex(all_dates)
            # Crypto may have gaps due to exchange issues; ffill as backup
            crypto_reindexed = crypto_reindexed.ffill()
            result_frames.append(crypto_reindexed)

        # Merge horizontally
        if len(result_frames) == 1:
            unified = result_frames[0]
        else:
            unified = pd.concat(result_frames, axis=1)

        # Drop any leading NaN rows (before first valid data)
        unified = unified.dropna(how="all")

        unified.index.name = "date"
        return unified

    def _get_unified_date_range(self) -> pd.DatetimeIndex:
        """Build a complete daily date range spanning all ingested data."""
        min_dates: list[pd.Timestamp] = []
        max_dates: list[pd.Timestamp] = []

        if self._stock_data is not None:
            min_dates.append(self._stock_data.index.min())
            max_dates.append(self._stock_data.index.max())

        if self._crypto_data is not None:
            min_dates.append(self._crypto_data.index.min())
            max_dates.append(self._crypto_data.index.max())

        start = min(min_dates)
        end = max(max_dates)

        return pd.date_range(start=start, end=end, freq="D")

    def get_tickers(self) -> Dict[str, list[str]]:
        """Return dict of tickers by asset class."""
        result: Dict[str, list[str]] = {"equity": [], "crypto": []}

        if self._stock_data is not None:
            result["equity"] = list(self._stock_data.columns.get_level_values(0).unique())

        if self._crypto_data is not None:
            result["crypto"] = list(self._crypto_data.columns.get_level_values(0).unique())

        return result


if __name__ == "__main__":
    # Smoke test with loader
    from loader import MarketDataLoader

    loader = MarketDataLoader(lookback_days=30)
    stocks = loader.fetch_data(["AAPL"], "equity")
    crypto = loader.fetch_data(["BTC-USD"], "crypto")

    normalizer = HybridDataNormalizer()
    normalizer.ingest(stocks, "equity")
    normalizer.ingest(crypto, "crypto")

    unified = normalizer.normalize()
    print(f"Unified Shape: {unified.shape}")
    print(f"Date Range: {unified.index.min()} to {unified.index.max()}")
    print(f"Sample weekend row:\n{unified.loc[unified.index.dayofweek == 5].head(1)}")
