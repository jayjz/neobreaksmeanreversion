"""
Hybrid Trader - Main Entry Point

Usage:
    python -m src.main              # Run trading loop
    python -m src.main --dry-run    # Test without placing orders
    python -m src.main --once       # Run single cycle then exit
    python -m src.main --json-logs  # Output JSON formatted logs
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from decimal import Decimal
from typing import Dict, List, Optional

import backtrader as bt

from .config import load_config, TradingConfig
from .data.loader import MarketDataLoader
from .data.normalizer import HybridDataNormalizer
from .strategies import EquityReversal, CryptoReversal, setup_cerebro_with_feeds
from .execution import AlpacaExecutor, OrderRouter, SignalConfig, OrderResult
from .execution.models import OrderSideEnum
from .utils.logger import setup_logging, TradeLogger
from .health import HealthMonitor


def run_cycle(
    config: TradingConfig,
    router: OrderRouter,
    trade_logger: TradeLogger,
    dry_run: bool = False,
) -> Dict[str, float]:
    """
    Execute one trading cycle: Fetch -> Analyze -> Signal -> Execute.

    Args:
        config: Trading configuration
        router: OrderRouter for order execution
        trade_logger: TradeLogger for audit trail
        dry_run: If True, log signals but don't place orders

    Returns:
        Dict of signals generated this cycle
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting trading cycle...")

    # Step 1: Fetch data
    loader = MarketDataLoader(lookback_days=config.lookback_days)

    equity_data = None
    crypto_data = None

    if config.equity_tickers:
        logger.info(f"Fetching equity data: {config.equity_tickers}")
        equity_data = loader.fetch_data(config.equity_tickers, "equity")

    if config.crypto_tickers:
        logger.info(f"Fetching crypto data: {config.crypto_tickers}")
        crypto_data = loader.fetch_data(config.crypto_tickers, "crypto")

    # Step 2: Normalize
    normalizer = HybridDataNormalizer()
    if equity_data is not None:
        normalizer.ingest(equity_data, "equity")
    if crypto_data is not None:
        normalizer.ingest(crypto_data, "crypto")

    unified_df = normalizer.normalize()
    logger.info(
        f"Normalized data: {unified_df.shape[0]} bars, {unified_df.shape[1]} columns"
    )

    # Step 3: Setup Backtrader
    cerebro = bt.Cerebro()

    # Add strategy based on what tickers we have
    if config.equity_tickers:
        cerebro.addstrategy(EquityReversal)
    # Note: For crypto-only, would use CryptoReversal instead

    # Add data feeds
    setup_cerebro_with_feeds(
        cerebro,
        unified_df,
        config.equity_tickers,
        config.crypto_tickers,
    )

    # Step 4: Run analysis (not live trading via backtrader)
    cerebro.broker.setcash(100000)  # Dummy value for analysis
    results = cerebro.run()

    # Step 5: Extract signals from strategy
    if not results:
        logger.warning("No strategy results returned")
        return {}

    strategy = results[0]
    signals = getattr(strategy, "last_signals", {})

    if not signals:
        logger.info("No signals generated this cycle")
        return {}

    logger.info(f"Generated signals: {signals}")

    # Step 6: Execute via router
    if dry_run:
        logger.info("DRY RUN - logging signals only")
        router.execute_signals(signals, dry_run=True)
    else:
        orders = router.execute_signals(signals)
        logger.info(f"Executed {len(orders)} orders")

        # Log executed orders to trade audit trail
        for order in orders:
            if order.filled_qty > 0:
                trade_logger.log_order(order, OrderSideEnum.BUY)

    return signals


def create_price_fetcher(
    loader: MarketDataLoader, tickers: list[str]
) -> callable:  # type: ignore[valid-type]
    """
    Create price fetcher function for OrderRouter.

    Uses the MarketDataLoader to fetch latest prices.
    Results are cached to avoid redundant API calls.

    Args:
        loader: MarketDataLoader instance
        tickers: List of tickers to prefetch

    Returns:
        Callable that returns current price for a symbol
    """
    logger = logging.getLogger(__name__)
    cache: Dict[str, Decimal] = {}

    def fetcher(symbol: str) -> Optional[Decimal]:
        if symbol not in cache:
            try:
                asset_class = "crypto" if "-USD" in symbol else "equity"
                df = loader.fetch_data([symbol], asset_class)
                # Get latest close price
                if symbol in df.columns.get_level_values(0):
                    close_col = df[symbol]["Close"]
                    if not close_col.empty:
                        cache[symbol] = Decimal(str(close_col.iloc[-1]))
            except Exception as e:
                logger.warning(f"Failed to fetch price for {symbol}: {e}")
        return cache.get(symbol, Decimal("0"))

    return fetcher


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hybrid Trader")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: generate signals but don't place orders",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single cycle and exit",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Output logs in JSON format (for observability platforms)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path",
    )
    parser.add_argument(
        "--trade-log",
        type=str,
        default="trades.csv",
        help="Path to trade audit trail CSV (default: trades.csv)",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level=logging.INFO,
        json_format=args.json_logs,
        log_file=args.log_file,
    )
    logger = logging.getLogger(__name__)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()

    mode = "PAPER" if config.paper else "LIVE"
    logger.info(
        "Startup",
        extra={
            "trading_mode": mode,
            "equity_tickers": config.equity_tickers,
            "crypto_tickers": config.crypto_tickers,
            "dry_run": args.dry_run,
        },
    )

    # Initialize trade logger
    trade_logger = TradeLogger(filepath=args.trade_log)
    logger.info(f"Trade audit trail: {trade_logger.filepath}")

    # Initialize executor
    executor = AlpacaExecutor(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        paper=config.paper,
        live_trading_enabled=config.live_trading_enabled,
        max_order_value=config.max_order_value,
    )

    # Initialize health monitor
    health_monitor = HealthMonitor(executor=executor)

    # Verify API connectivity on startup
    logger.info("Checking API connectivity...")
    if not health_monitor.check_api():
        logger.error("API health check failed on startup")
        if not args.dry_run:
            raise RuntimeError("Cannot start trading: API unreachable")
        logger.warning("Continuing in dry-run mode despite API failure")

    # Initialize router with price fetcher
    loader = MarketDataLoader(lookback_days=1)  # For price fetching
    all_tickers = config.equity_tickers + config.crypto_tickers
    price_fetcher = create_price_fetcher(loader, all_tickers)

    signal_config = SignalConfig()
    router = OrderRouter(executor, config=signal_config, price_fetcher=price_fetcher)

    # Main loop
    logger.info("Starting trading loop...")
    cycle_count = 0

    while True:
        cycle_count += 1
        logger.info(f"Cycle {cycle_count} starting")

        try:
            # Update heartbeat at start of each cycle
            health_monitor.heartbeat()

            # Run trading cycle
            signals = run_cycle(config, router, trade_logger, dry_run=args.dry_run)

            # Update heartbeat after successful cycle
            health_monitor.heartbeat()

            logger.info(
                "Cycle completed",
                extra={
                    "cycle": cycle_count,
                    "signals_count": len(signals),
                    "health_status": health_monitor.get_status(),
                },
            )

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            break
        except Exception as e:
            logger.exception(f"Cycle error: {e}")
            # Don't crash - log and continue

        if args.once:
            logger.info("Single cycle complete, exiting")
            break

        logger.info(
            f"Sleeping {config.cycle_interval_minutes} minutes until next cycle..."
        )
        time.sleep(config.cycle_interval_minutes * 60)

    # Cleanup
    logger.info(
        "Trader shutdown complete",
        extra={
            "total_cycles": cycle_count,
            "health_status": health_monitor.get_status(),
        },
    )


if __name__ == "__main__":
    main()
