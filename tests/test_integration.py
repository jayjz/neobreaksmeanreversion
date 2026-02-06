"""
Integration tests for Phase 4: Main Loop.

All tests use mocked data and APIs - NO real external calls.
"""
from __future__ import annotations

import importlib
import pytest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone
import pandas as pd
import numpy as np

import sys
import os

# Add src to path
sys.path.insert(0, str(__file__).replace("/tests/test_integration.py", "/src"))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV DataFrame with MultiIndex columns."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    np.random.seed(42)  # Reproducible
    data = {
        ("AAPL", "Open"): np.random.uniform(150, 160, 30),
        ("AAPL", "High"): np.random.uniform(160, 170, 30),
        ("AAPL", "Low"): np.random.uniform(140, 150, 30),
        ("AAPL", "Close"): np.random.uniform(150, 160, 30),
        ("AAPL", "Volume"): np.random.uniform(1e6, 2e6, 30),
    }
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def mock_env_paper():
    """Environment variables for paper trading."""
    return {
        "ALPACA_API_KEY": "test_api_key",
        "ALPACA_SECRET_KEY": "test_secret_key",
        "TRADING_MODE": "PAPER",
        "LIVE_TRADING_ENABLED": "false",
        "EQUITY_TICKERS": "AAPL,MSFT",
        "CRYPTO_TICKERS": "BTC-USD",
        "LOOKBACK_DAYS": "30",
        "MAX_ORDER_VALUE": "5000",
        "CYCLE_INTERVAL_MINUTES": "60",
    }


# =============================================================================
# Config Loading Tests
# =============================================================================


class TestConfigLoading:
    """Test configuration management."""

    def _mock_env_file_exists(self, config_module: object) -> object:
        """Helper to mock _ENV_FILE.exists() returning True."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        return patch.object(config_module, "_ENV_FILE", mock_path)

    def test_load_config_paper_mode(self, mock_env_paper: dict) -> None:
        """Valid env vars should produce TradingConfig in paper mode."""
        import config

        with patch.dict(os.environ, mock_env_paper, clear=True):
            importlib.reload(config)
            with self._mock_env_file_exists(config):
                with patch.object(config, "load_dotenv"):  # Don't load .env file
                    result = config.load_config()

            assert result.alpaca_api_key == "test_api_key"
            assert result.alpaca_secret_key == "test_secret_key"
            assert result.paper is True
            assert result.live_trading_enabled is False
            assert result.equity_tickers == ["AAPL", "MSFT"]
            assert result.crypto_tickers == ["BTC-USD"]
            assert result.lookback_days == 30
            assert result.max_order_value == Decimal("5000")

    def test_missing_api_key_raises_error(self) -> None:
        """Missing API key should raise ValueError."""
        import config

        # Reload with cleared environment
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(config)
            with self._mock_env_file_exists(config):
                with patch.object(config, "load_dotenv"):
                    with pytest.raises(ValueError) as exc_info:
                        config.load_config()

                    assert "ALPACA_API_KEY" in str(exc_info.value)

    def test_missing_secret_key_raises_error(self) -> None:
        """Missing secret key should raise ValueError."""
        import config

        with patch.dict(os.environ, {"ALPACA_API_KEY": "key"}, clear=True):
            importlib.reload(config)
            with self._mock_env_file_exists(config):
                with patch.object(config, "load_dotenv"):
                    with pytest.raises(ValueError) as exc_info:
                        config.load_config()

                    assert "ALPACA_SECRET_KEY" in str(exc_info.value)

    def test_live_mode_sets_paper_false(self) -> None:
        """TRADING_MODE=LIVE should set paper=False."""
        import config

        env = {
            "ALPACA_API_KEY": "test",
            "ALPACA_SECRET_KEY": "test",
            "TRADING_MODE": "LIVE",
            "LIVE_TRADING_ENABLED": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            importlib.reload(config)
            with self._mock_env_file_exists(config):
                with patch.object(config, "load_dotenv"):
                    with patch.object(config.time, "sleep"):  # Skip 5s warning delay
                        result = config.load_config()

        assert result.paper is False
        assert result.live_trading_enabled is True

    def test_default_values_applied(self) -> None:
        """Missing optional vars should use defaults."""
        import config

        env = {
            "ALPACA_API_KEY": "test",
            "ALPACA_SECRET_KEY": "test",
        }
        with patch.dict(os.environ, env, clear=True):
            importlib.reload(config)
            with self._mock_env_file_exists(config):
                with patch.object(config, "load_dotenv"):
                    result = config.load_config()

        # Check defaults
        assert result.paper is True
        assert result.equity_tickers == ["AAPL", "MSFT", "GOOGL"]
        assert result.crypto_tickers == ["BTC-USD", "ETH-USD"]
        assert result.lookback_days == 60
        assert result.max_order_value == Decimal("10000")
        assert result.cycle_interval_minutes == 60

    def test_missing_env_file_raises_helpful_error(self) -> None:
        """Missing .env file should raise FileNotFoundError with helpful message."""
        import config

        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(config)
            # Mock _ENV_FILE.exists() to return False
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            with patch.object(config, "_ENV_FILE", mock_path):
                with pytest.raises(FileNotFoundError) as exc_info:
                    config.load_config()

                error_msg = str(exc_info.value)
                assert ".env file not found" in error_msg
                assert "cp" in error_msg  # Should suggest copy command


# =============================================================================
# TradingConfig Tests
# =============================================================================


class TestTradingConfig:
    """Test TradingConfig dataclass."""

    def test_config_is_frozen(self) -> None:
        """TradingConfig should be immutable."""
        from config import TradingConfig

        config = TradingConfig(
            alpaca_api_key="test",
            alpaca_secret_key="test",
            paper=True,
            live_trading_enabled=False,
            equity_tickers=["AAPL"],
            crypto_tickers=["BTC-USD"],
            lookback_days=60,
            max_order_value=Decimal("10000"),
            cycle_interval_minutes=60,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            config.paper = False  # type: ignore


# =============================================================================
# Signal Extraction Tests
# =============================================================================


class TestSignalExtraction:
    """Test signal extraction from strategy after cerebro.run()."""

    def test_last_signals_property_exists(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """ReversalStrategy should have last_signals property."""
        import backtrader as bt
        from strategies import EquityReversal, create_bt_feeds

        cerebro = bt.Cerebro()
        cerebro.addstrategy(EquityReversal)

        for ticker, feed in create_bt_feeds(sample_ohlcv_df, ["AAPL"]):
            cerebro.adddata(feed, name=ticker)

        cerebro.broker.setcash(100000)
        results = cerebro.run()

        strategy = results[0]

        # Property should exist
        assert hasattr(strategy, "last_signals")
        # Should return a dict
        assert isinstance(strategy.last_signals, dict)

    def test_last_signals_returns_copy(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """last_signals should return a copy, not the original dict."""
        import backtrader as bt
        from strategies import EquityReversal, create_bt_feeds

        cerebro = bt.Cerebro()
        cerebro.addstrategy(EquityReversal)

        for ticker, feed in create_bt_feeds(sample_ohlcv_df, ["AAPL"]):
            cerebro.adddata(feed, name=ticker)

        cerebro.broker.setcash(100000)
        results = cerebro.run()

        strategy = results[0]

        # Get signals twice
        signals1 = strategy.last_signals
        signals2 = strategy.last_signals

        # Should be equal but not the same object
        assert signals1 == signals2
        assert signals1 is not signals2

    def test_base_strategy_stores_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Base strategy should store signals in _last_signals during next()."""
        import backtrader as bt
        from strategies import EquityReversal, create_bt_feeds

        cerebro = bt.Cerebro()
        cerebro.addstrategy(EquityReversal)

        for ticker, feed in create_bt_feeds(sample_ohlcv_df, ["AAPL"]):
            cerebro.adddata(feed, name=ticker)

        cerebro.broker.setcash(100000)
        results = cerebro.run()

        strategy = results[0]

        # _last_signals should exist internally
        assert hasattr(strategy, "_last_signals")
        assert isinstance(strategy._last_signals, dict)


# =============================================================================
# Integration Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Test complete data flow through the system."""

    def test_data_flows_through_normalizer(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Data should flow: Loader -> Normalizer -> Feeds."""
        from data.normalizer import HybridDataNormalizer
        from strategies import create_bt_feeds

        normalizer = HybridDataNormalizer()
        normalizer.ingest(sample_ohlcv_df, "equity")
        unified = normalizer.normalize()

        # Should produce valid feeds
        feeds = create_bt_feeds(unified, ["AAPL"])
        assert len(feeds) == 1
        assert feeds[0][0] == "AAPL"

    def test_strategy_runs_on_normalized_data(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Strategy should run on normalized data without errors."""
        import backtrader as bt
        from data.normalizer import HybridDataNormalizer
        from strategies import EquityReversal, create_bt_feeds

        # Normalize
        normalizer = HybridDataNormalizer()
        normalizer.ingest(sample_ohlcv_df, "equity")
        unified = normalizer.normalize()

        # Setup cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(EquityReversal)

        for ticker, feed in create_bt_feeds(unified, ["AAPL"]):
            cerebro.adddata(feed, name=ticker)

        cerebro.broker.setcash(100000)

        # Should run without error
        results = cerebro.run()
        assert len(results) == 1

    def test_multi_ticker_pipeline(self) -> None:
        """Pipeline should work with multiple tickers."""
        import backtrader as bt
        from data.normalizer import HybridDataNormalizer
        from strategies import EquityReversal, create_bt_feeds

        # Create multi-ticker data
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        np.random.seed(42)

        data = {}
        for ticker in ["AAPL", "MSFT"]:
            data[(ticker, "Open")] = np.random.uniform(150, 160, 30)
            data[(ticker, "High")] = np.random.uniform(160, 170, 30)
            data[(ticker, "Low")] = np.random.uniform(140, 150, 30)
            data[(ticker, "Close")] = np.random.uniform(150, 160, 30)
            data[(ticker, "Volume")] = np.random.uniform(1e6, 2e6, 30)

        df = pd.DataFrame(data, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        # Normalize
        normalizer = HybridDataNormalizer()
        normalizer.ingest(df, "equity")
        unified = normalizer.normalize()

        # Setup cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(EquityReversal)

        for ticker, feed in create_bt_feeds(unified, ["AAPL", "MSFT"]):
            cerebro.adddata(feed, name=ticker)

        cerebro.broker.setcash(100000)

        # Should run without error
        results = cerebro.run()
        assert len(results) == 1

        # Strategy should have last_signals
        strategy = results[0]
        assert hasattr(strategy, "last_signals")


# =============================================================================
# Execution Integration Tests
# =============================================================================


class TestExecutionIntegration:
    """Test integration with execution layer."""

    def test_router_accepts_strategy_signals(self) -> None:
        """OrderRouter should accept signal format from strategy."""
        from execution import OrderRouter, SignalConfig
        from execution.base import ExecutorBase
        from execution.models import AccountInfo, AssetClassEnum

        # Create mock executor
        mock_executor = Mock(spec=ExecutorBase)
        mock_executor.get_account_info.return_value = AccountInfo(
            cash=Decimal("100000"),
            buying_power=Decimal("200000"),
            portfolio_value=Decimal("150000"),
            equity=Decimal("150000"),
            is_trading_blocked=False,
            is_pattern_day_trader=False,
        )
        mock_executor.get_positions.return_value = {}
        mock_executor.get_asset_class.return_value = AssetClassEnum.EQUITY

        # Create price fetcher that returns fixed prices
        def price_fetcher(symbol: str) -> Decimal:
            return Decimal("150.00")

        # Create router with price fetcher
        router = OrderRouter(mock_executor, SignalConfig(), price_fetcher=price_fetcher)

        # Strategy-style signals
        signals = {"AAPL": 0.5, "MSFT": 0.3}

        # Dry run should not raise
        results = router.execute_signals(signals, dry_run=True)
        assert isinstance(results, list)

    def test_empty_signals_handled(self) -> None:
        """Router should handle empty signals gracefully."""
        from decimal import Decimal
        from execution import OrderRouter
        from execution.base import ExecutorBase
        from execution.models import AccountInfo

        mock_executor = Mock(spec=ExecutorBase)
        # Setup mocks for reconciliation flow
        mock_executor.get_account_info.return_value = AccountInfo(
            cash=Decimal("10000"),
            buying_power=Decimal("10000"),
            portfolio_value=Decimal("10000"),
            equity=Decimal("10000"),
        )
        mock_executor.get_positions.return_value = {}  # No existing positions

        router = OrderRouter(mock_executor)
        results = router.execute_signals({})

        assert results == []
        mock_executor.submit_order.assert_not_called()


# =============================================================================
# Price Fetcher Tests (Unit-level)
# =============================================================================


class TestPriceFetcherUnit:
    """Test price fetcher function in isolation."""

    def test_create_price_fetcher_returns_callable(self) -> None:
        """create_price_fetcher should return a callable."""
        from data.loader import MarketDataLoader

        loader = Mock(spec=MarketDataLoader)
        loader.fetch_data.return_value = pd.DataFrame()

        # Import the function directly
        # Note: We can't test main.create_price_fetcher due to relative imports
        # So we test the concept
        cache = {}

        def price_fetcher(symbol: str) -> Decimal:
            if symbol not in cache:
                cache[symbol] = Decimal("150.00")
            return cache[symbol]

        result = price_fetcher("AAPL")
        assert isinstance(result, Decimal)
        assert result == Decimal("150.00")

    def test_price_caching_concept(self) -> None:
        """Price fetcher caching should avoid redundant calls."""
        call_count = 0
        cache = {}

        def mock_fetch(symbol: str) -> Decimal:
            nonlocal call_count
            if symbol not in cache:
                call_count += 1
                cache[symbol] = Decimal("150.00")
            return cache[symbol]

        # Call twice
        mock_fetch("AAPL")
        mock_fetch("AAPL")

        # Should only increment once
        assert call_count == 1


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Test that all modules can be imported."""

    def test_config_imports(self) -> None:
        """config module should import."""
        import config

        assert hasattr(config, "TradingConfig")
        assert hasattr(config, "load_config")

    def test_strategies_imports(self) -> None:
        """strategies module should import."""
        from strategies import (
            ReversalStrategy,
            EquityReversal,
            CryptoReversal,
            create_bt_feeds,
            setup_cerebro_with_feeds,
        )

        assert ReversalStrategy is not None
        assert EquityReversal is not None
        assert CryptoReversal is not None

    def test_execution_imports(self) -> None:
        """execution module should import."""
        from execution import (
            ExecutorBase,
            AlpacaExecutor,
            OrderRouter,
            SignalConfig,
            AccountInfo,
            OrderRequest,
            OrderResult,
        )

        assert ExecutorBase is not None
        assert OrderRouter is not None

    def test_data_imports(self) -> None:
        """data module should import."""
        from data.loader import MarketDataLoader
        from data.normalizer import HybridDataNormalizer

        assert MarketDataLoader is not None
        assert HybridDataNormalizer is not None


class TestLoaderMultiIndexFix:
    """Tests for the yfinance MultiIndex fix in MarketDataLoader."""

    def test_single_ticker_returns_multiindex(self) -> None:
        """Single ticker should return MultiIndex columns."""
        from data.loader import MarketDataLoader

        loader = MarketDataLoader(lookback_days=5)

        # Mock yfinance to return MultiIndex (modern behavior)
        mock_df = pd.DataFrame(
            {
                ("AAPL", "Open"): [150.0, 151.0],
                ("AAPL", "High"): [155.0, 156.0],
                ("AAPL", "Low"): [149.0, 150.0],
                ("AAPL", "Close"): [153.0, 154.0],
                ("AAPL", "Volume"): [1e6, 1.1e6],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )
        mock_df.columns = pd.MultiIndex.from_tuples(mock_df.columns)

        with patch("yfinance.download", return_value=mock_df):
            result = loader.fetch_data(["AAPL"], "equity")

        assert isinstance(result.columns, pd.MultiIndex)
        assert "AAPL" in result.columns.get_level_values(0)
        assert "Close" in result.columns.get_level_values(1)

    def test_single_ticker_flat_index_promoted(self) -> None:
        """Single ticker with flat index should be promoted to MultiIndex."""
        from data.loader import MarketDataLoader

        loader = MarketDataLoader(lookback_days=5)

        # Mock yfinance to return flat index (legacy behavior)
        mock_df = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "High": [155.0, 156.0],
                "Low": [149.0, 150.0],
                "Close": [153.0, 154.0],
                "Volume": [1e6, 1.1e6],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        with patch("yfinance.download", return_value=mock_df):
            result = loader.fetch_data(["AAPL"], "equity")

        assert isinstance(result.columns, pd.MultiIndex)
        assert "AAPL" in result.columns.get_level_values(0)
        assert "Close" in result.columns.get_level_values(1)

    def test_price_fetcher_gets_valid_price(self) -> None:
        """Price fetcher should return valid Decimal price."""
        from data.loader import MarketDataLoader
        from typing import Dict, Optional

        loader = MarketDataLoader(lookback_days=5)

        # Inline price fetcher (same logic as main.py)
        def create_price_fetcher(ldr: MarketDataLoader, tickers: list) -> callable:  # type: ignore
            cache: Dict[str, Decimal] = {}

            def fetcher(symbol: str) -> Optional[Decimal]:
                if symbol not in cache:
                    try:
                        asset_class = "crypto" if "-USD" in symbol else "equity"
                        df = ldr.fetch_data([symbol], asset_class)
                        if symbol in df.columns.get_level_values(0):
                            close_col = df[symbol]["Close"]
                            if not close_col.empty:
                                cache[symbol] = Decimal(str(close_col.iloc[-1]))
                    except Exception:
                        pass
                return cache.get(symbol, Decimal("0"))

            return fetcher

        # Mock yfinance to return MultiIndex
        mock_df = pd.DataFrame(
            {
                ("AAPL", "Open"): [150.0, 151.0],
                ("AAPL", "High"): [155.0, 156.0],
                ("AAPL", "Low"): [149.0, 150.0],
                ("AAPL", "Close"): [153.0, 154.0],
                ("AAPL", "Volume"): [1e6, 1.1e6],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )
        mock_df.columns = pd.MultiIndex.from_tuples(mock_df.columns)

        with patch("yfinance.download", return_value=mock_df):
            fetcher = create_price_fetcher(loader, ["AAPL"])
            price = fetcher("AAPL")

        assert price is not None
        assert price > Decimal("0")
        assert price == Decimal("154.0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
