"""
Execution layer for live/paper trading.

This module provides the bridge between backtrader strategies and
live execution via brokers like Alpaca.

Safety:
    - AlpacaExecutor defaults to paper=True
    - Live trading requires explicit live_trading_enabled=True
    - All tests use mocked APIs (no real orders)
"""
from .base import ExecutorBase, ExecutorError, SafetyError
from .models import (
    AccountInfo,
    PositionInfo,
    OrderRequest,
    OrderResult,
    OrderSideEnum,
    OrderTypeEnum,
    OrderStatusEnum,
    AssetClassEnum,
)
from .alpaca import AlpacaExecutor
from .router import OrderRouter, SignalConfig, PendingOrder

__all__ = [
    # Base classes
    "ExecutorBase",
    "ExecutorError",
    "SafetyError",
    # Data models
    "AccountInfo",
    "PositionInfo",
    "OrderRequest",
    "OrderResult",
    "OrderSideEnum",
    "OrderTypeEnum",
    "OrderStatusEnum",
    "AssetClassEnum",
    # Implementations
    "AlpacaExecutor",
    "OrderRouter",
    "SignalConfig",
    "PendingOrder",
]
