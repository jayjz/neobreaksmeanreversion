"""
Strategy Engine for Hybrid Reversal Trading System.

Strategies:
- ReversalStrategy: Abstract base with risk management
- EquityReversal: S&P 500 mean reversion (bottom decile weekly)
- CryptoReversal: Volatility-gated crypto reversion (ADX < 25)

Utilities:
- create_bt_feeds: Convert normalized DataFrame to backtrader feeds
- setup_cerebro_with_feeds: Helper to configure Cerebro
"""
from .base import ReversalStrategy, PositionTracker
from .equity import EquityReversal
from .crypto import CryptoReversal
from .feed_adapter import create_bt_feeds, setup_cerebro_with_feeds

__all__ = [
    "ReversalStrategy",
    "PositionTracker",
    "EquityReversal",
    "CryptoReversal",
    "create_bt_feeds",
    "setup_cerebro_with_feeds",
]
