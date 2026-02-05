"""
Configuration management for Hybrid Trader.

Loads environment variables from .env file with strict validation.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import List

from dotenv import load_dotenv


@dataclass(frozen=True)
class TradingConfig:
    """Immutable trading configuration."""

    alpaca_api_key: str
    alpaca_secret_key: str
    paper: bool  # True = paper trading
    live_trading_enabled: bool  # Safety flag required for live

    # Trading parameters
    equity_tickers: List[str]
    crypto_tickers: List[str]
    lookback_days: int
    max_order_value: Decimal

    # Cycle control
    cycle_interval_minutes: int


def load_config() -> TradingConfig:
    """
    Load configuration from .env file with strict validation.

    Environment Variables:
        ALPACA_API_KEY: Required. Your Alpaca API key.
        ALPACA_SECRET_KEY: Required. Your Alpaca secret key.
        TRADING_MODE: "PAPER" (default) or "LIVE".
        LIVE_TRADING_ENABLED: "true" to enable live trading (safety flag).
        EQUITY_TICKERS: Comma-separated list of equity symbols.
        CRYPTO_TICKERS: Comma-separated list of crypto symbols.
        LOOKBACK_DAYS: Historical data lookback (default: 60).
        MAX_ORDER_VALUE: Maximum single order value (default: 10000).
        CYCLE_INTERVAL_MINUTES: Minutes between cycles (default: 60).

    Returns:
        TradingConfig with validated settings.

    Raises:
        ValueError: If required environment variables are missing.
    """
    load_dotenv()

    # Required keys (fail if missing)
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY are required. "
            "Set them in your .env file."
        )

    # TRADING_MODE: "PAPER" (default) or "LIVE"
    trading_mode = os.environ.get("TRADING_MODE", "PAPER").upper()
    paper = trading_mode != "LIVE"

    # SAFETY: Live trading requires explicit LIVE_TRADING_ENABLED=true
    live_enabled = os.environ.get("LIVE_TRADING_ENABLED", "false").lower() == "true"

    if not paper:
        _print_live_warning_banner()

    return TradingConfig(
        alpaca_api_key=api_key,
        alpaca_secret_key=secret_key,
        paper=paper,
        live_trading_enabled=live_enabled,
        equity_tickers=_parse_list("EQUITY_TICKERS", ["AAPL", "MSFT", "GOOGL"]),
        crypto_tickers=_parse_list("CRYPTO_TICKERS", ["BTC-USD", "ETH-USD"]),
        lookback_days=int(os.environ.get("LOOKBACK_DAYS", "60")),
        max_order_value=Decimal(os.environ.get("MAX_ORDER_VALUE", "10000")),
        cycle_interval_minutes=int(os.environ.get("CYCLE_INTERVAL_MINUTES", "60")),
    )


def _parse_list(env_var: str, default: List[str]) -> List[str]:
    """Parse comma-separated list from environment variable."""
    val = os.environ.get(env_var)
    if not val:
        return default
    return [s.strip() for s in val.split(",") if s.strip()]


def _print_live_warning_banner() -> None:
    """Print high-visibility warning for live trading mode."""
    banner = """
================================================================================
                         LIVE TRADING MODE ENABLED
================================================================================

  Real money is at risk. Orders will execute against your actual Alpaca account.

  Please verify:
    - ALPACA_API_KEY is for the correct account
    - LIVE_TRADING_ENABLED is intentionally set to "true"
    - You understand the risks of automated trading

  Press Ctrl+C within 5 seconds to abort...

================================================================================
"""
    print(banner)
    time.sleep(5)
    print("Proceeding with live trading...\n")
