"""
Tests for HybridDataNormalizer.
Verifies weekend alignment logic for hybrid asset normalization.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).replace("/tests/test_normalizer.py", "/src/data"))

from normalizer import HybridDataNormalizer


def create_mock_stock_data() -> pd.DataFrame:
    """
    Create mock stock data (M-F only, no weekends).
    Week of 2024-01-08 to 2024-01-12 (Mon-Fri).
    """
    dates = pd.to_datetime([
        "2024-01-08",  # Monday
        "2024-01-09",  # Tuesday
        "2024-01-10",  # Wednesday
        "2024-01-11",  # Thursday
        "2024-01-12",  # Friday
        "2024-01-15",  # Monday (next week)
        "2024-01-16",  # Tuesday
    ])

    # AAPL prices (incrementing to make ffill obvious)
    aapl_data = {
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 107.0, 108.0],
        "High": [101.0, 102.0, 103.0, 104.0, 105.0, 108.0, 109.0],
        "Low": [99.0, 100.0, 101.0, 102.0, 103.0, 106.0, 107.0],
        "Close": [100.5, 101.5, 102.5, 103.5, 104.5, 107.5, 108.5],
        "Volume": [1000, 1100, 1200, 1300, 1400, 1700, 1800],
    }

    df = pd.DataFrame(aapl_data, index=dates)
    df.columns = pd.MultiIndex.from_product([["AAPL"], df.columns])
    df.index.name = "date"
    return df


def create_mock_crypto_data() -> pd.DataFrame:
    """
    Create mock crypto data (all days including weekends).
    Same period as stock data but includes Sat/Sun.
    """
    dates = pd.to_datetime([
        "2024-01-08",  # Monday
        "2024-01-09",  # Tuesday
        "2024-01-10",  # Wednesday
        "2024-01-11",  # Thursday
        "2024-01-12",  # Friday
        "2024-01-13",  # Saturday
        "2024-01-14",  # Sunday
        "2024-01-15",  # Monday
        "2024-01-16",  # Tuesday
    ])

    # BTC-USD prices (different values on weekend to prove crypto trades)
    btc_data = {
        "Open": [42000.0, 42100.0, 42200.0, 42300.0, 42400.0, 42500.0, 42600.0, 42700.0, 42800.0],
        "High": [42500.0, 42600.0, 42700.0, 42800.0, 42900.0, 43000.0, 43100.0, 43200.0, 43300.0],
        "Low": [41500.0, 41600.0, 41700.0, 41800.0, 41900.0, 42000.0, 42100.0, 42200.0, 42300.0],
        "Close": [42100.0, 42200.0, 42300.0, 42400.0, 42500.0, 42600.0, 42700.0, 42800.0, 42900.0],
        "Volume": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000],
    }

    df = pd.DataFrame(btc_data, index=dates)
    df.columns = pd.MultiIndex.from_product([["BTC-USD"], df.columns])
    df.index.name = "date"
    return df


class TestHybridDataNormalizer:
    """Test suite for HybridDataNormalizer."""

    @pytest.fixture
    def normalizer(self) -> HybridDataNormalizer:
        """Create a normalizer with mock data ingested."""
        norm = HybridDataNormalizer()
        norm.ingest(create_mock_stock_data(), "equity")
        norm.ingest(create_mock_crypto_data(), "crypto")
        return norm

    @pytest.fixture
    def unified_df(self, normalizer: HybridDataNormalizer) -> pd.DataFrame:
        """Get the normalized unified DataFrame."""
        return normalizer.normalize()

    def test_saturday_row_exists(self, unified_df: pd.DataFrame) -> None:
        """Verify that Saturday rows are present in the unified index."""
        saturday_mask = unified_df.index.dayofweek == 5  # Saturday = 5
        saturday_rows = unified_df[saturday_mask]

        assert len(saturday_rows) > 0, "No Saturday rows found in unified DataFrame"
        assert pd.Timestamp("2024-01-13") in unified_df.index, "2024-01-13 (Saturday) not in index"

    def test_sunday_row_exists(self, unified_df: pd.DataFrame) -> None:
        """Verify that Sunday rows are present in the unified index."""
        sunday_mask = unified_df.index.dayofweek == 6  # Sunday = 6
        sunday_rows = unified_df[sunday_mask]

        assert len(sunday_rows) > 0, "No Sunday rows found in unified DataFrame"
        assert pd.Timestamp("2024-01-14") in unified_df.index, "2024-01-14 (Sunday) not in index"

    def test_stock_close_saturday_equals_friday(self, unified_df: pd.DataFrame) -> None:
        """Stock Close on Saturday should equal Friday Close (forward-fill)."""
        friday = pd.Timestamp("2024-01-12")
        saturday = pd.Timestamp("2024-01-13")

        friday_close = unified_df.loc[friday, ("AAPL", "Close")]
        saturday_close = unified_df.loc[saturday, ("AAPL", "Close")]

        assert friday_close == saturday_close, (
            f"Stock ffill failed: Friday={friday_close}, Saturday={saturday_close}"
        )

    def test_stock_close_sunday_equals_friday(self, unified_df: pd.DataFrame) -> None:
        """Stock Close on Sunday should also equal Friday Close (forward-fill)."""
        friday = pd.Timestamp("2024-01-12")
        sunday = pd.Timestamp("2024-01-14")

        friday_close = unified_df.loc[friday, ("AAPL", "Close")]
        sunday_close = unified_df.loc[sunday, ("AAPL", "Close")]

        assert friday_close == sunday_close, (
            f"Stock ffill failed: Friday={friday_close}, Sunday={sunday_close}"
        )

    def test_crypto_close_saturday_not_equal_friday(self, unified_df: pd.DataFrame) -> None:
        """Crypto Close on Saturday should differ from Friday (crypto trades 24/7)."""
        friday = pd.Timestamp("2024-01-12")
        saturday = pd.Timestamp("2024-01-13")

        friday_close = unified_df.loc[friday, ("BTC-USD", "Close")]
        saturday_close = unified_df.loc[saturday, ("BTC-USD", "Close")]

        assert friday_close != saturday_close, (
            f"Crypto should have different weekend prices: Friday={friday_close}, Saturday={saturday_close}"
        )

    def test_crypto_close_sunday_not_equal_friday(self, unified_df: pd.DataFrame) -> None:
        """Crypto Close on Sunday should differ from Friday."""
        friday = pd.Timestamp("2024-01-12")
        sunday = pd.Timestamp("2024-01-14")

        friday_close = unified_df.loc[friday, ("BTC-USD", "Close")]
        sunday_close = unified_df.loc[sunday, ("BTC-USD", "Close")]

        assert friday_close != sunday_close, (
            f"Crypto should have different weekend prices: Friday={friday_close}, Sunday={sunday_close}"
        )

    def test_unified_index_is_continuous(self, unified_df: pd.DataFrame) -> None:
        """The unified index should have no gaps (all days present)."""
        expected_days = pd.date_range(
            start=unified_df.index.min(),
            end=unified_df.index.max(),
            freq="D"
        )

        assert len(unified_df.index) == len(expected_days), (
            f"Index has gaps: expected {len(expected_days)} days, got {len(unified_df.index)}"
        )

    def test_no_nan_values_after_normalization(self, unified_df: pd.DataFrame) -> None:
        """After ffill, there should be no NaN values in the data range."""
        # Check only after the first row (which may have leading NaNs before first valid data)
        nan_count = unified_df.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values after normalization"

    def test_get_tickers(self, normalizer: HybridDataNormalizer) -> None:
        """Verify get_tickers returns correct ticker lists."""
        tickers = normalizer.get_tickers()

        assert "equity" in tickers
        assert "crypto" in tickers
        assert "AAPL" in tickers["equity"]
        assert "BTC-USD" in tickers["crypto"]


class TestHybridDataNormalizerEdgeCases:
    """Test edge cases for HybridDataNormalizer."""

    def test_ingest_invalid_asset_class(self) -> None:
        """Should raise ValueError for unknown asset class."""
        normalizer = HybridDataNormalizer()
        df = create_mock_stock_data()

        with pytest.raises(ValueError, match="Unknown asset_class"):
            normalizer.ingest(df, "forex")  # type: ignore

    def test_normalize_without_ingest(self) -> None:
        """Should raise ValueError when normalize() called without data."""
        normalizer = HybridDataNormalizer()

        with pytest.raises(ValueError, match="No data ingested"):
            normalizer.normalize()

    def test_stock_only_normalization(self) -> None:
        """Normalizer should work with only stock data."""
        normalizer = HybridDataNormalizer()
        normalizer.ingest(create_mock_stock_data(), "equity")

        unified = normalizer.normalize()

        # Should have weekends filled
        assert pd.Timestamp("2024-01-13") in unified.index
        assert pd.Timestamp("2024-01-14") in unified.index

    def test_crypto_only_normalization(self) -> None:
        """Normalizer should work with only crypto data."""
        normalizer = HybridDataNormalizer()
        normalizer.ingest(create_mock_crypto_data(), "crypto")

        unified = normalizer.normalize()

        assert len(unified) > 0
        assert "BTC-USD" in unified.columns.get_level_values(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
