"""
Adapter to convert HybridDataNormalizer output to backtrader feeds.
"""
from __future__ import annotations

import backtrader as bt
import pandas as pd
from typing import List, Tuple


def create_bt_feeds(
    unified_df: pd.DataFrame,
    tickers: List[str],
) -> List[Tuple[str, bt.feeds.PandasData]]:
    """
    Convert unified MultiIndex DataFrame to individual backtrader feeds.

    Args:
        unified_df: DataFrame from HybridDataNormalizer.normalize()
                   Has MultiIndex columns (ticker, OHLCV)
        tickers: List of tickers to extract

    Returns:
        List of (ticker_name, PandasData) tuples ready for cerebro.adddata()
    """
    feeds: List[Tuple[str, bt.feeds.PandasData]] = []

    for ticker in tickers:
        if ticker not in unified_df.columns.get_level_values(0):
            continue

        # Extract single ticker data
        ticker_df = unified_df[ticker].copy()

        # Standardize column names for backtrader
        ticker_df.columns = [str(col).capitalize() for col in ticker_df.columns]

        # Ensure required columns exist
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in ticker_df.columns:
                # Try case-insensitive match
                for existing in ticker_df.columns:
                    if existing.lower() == col.lower():
                        ticker_df = ticker_df.rename(columns={existing: col})
                        break

        # Ensure datetime index
        if not isinstance(ticker_df.index, pd.DatetimeIndex):
            ticker_df.index = pd.to_datetime(ticker_df.index)

        # Create PandasData feed
        feed = bt.feeds.PandasData(
            dataname=ticker_df,
            datetime=None,  # Use index
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            openinterest=-1,  # Not available
        )

        feeds.append((ticker, feed))

    return feeds


def setup_cerebro_with_feeds(
    cerebro: bt.Cerebro,
    unified_df: pd.DataFrame,
    equity_tickers: List[str],
    crypto_tickers: List[str],
) -> None:
    """
    Add data feeds to cerebro with proper naming.

    Args:
        cerebro: Backtrader Cerebro instance
        unified_df: Normalized data from HybridDataNormalizer
        equity_tickers: List of equity ticker symbols
        crypto_tickers: List of crypto ticker symbols
    """
    # Add equity feeds
    for ticker, feed in create_bt_feeds(unified_df, equity_tickers):
        cerebro.adddata(feed, name=ticker)

    # Add crypto feeds
    for ticker, feed in create_bt_feeds(unified_df, crypto_tickers):
        cerebro.adddata(feed, name=ticker)
