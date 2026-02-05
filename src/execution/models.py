"""
Data transfer objects for execution layer.
Provides type-safe wrappers around broker-specific models.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Optional


class OrderSideEnum(Enum):
    """Order direction."""

    BUY = auto()
    SELL = auto()


class OrderTypeEnum(Enum):
    """Order type."""

    MARKET = auto()
    LIMIT = auto()
    STOP_LIMIT = auto()


class OrderStatusEnum(Enum):
    """Order execution status."""

    PENDING = auto()
    SUBMITTED = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class AssetClassEnum(Enum):
    """Asset classification."""

    EQUITY = auto()
    CRYPTO = auto()


@dataclass(frozen=True)
class AccountInfo:
    """
    Snapshot of account state.

    Attributes:
        cash: Available cash balance
        buying_power: Total buying power (including margin)
        portfolio_value: Total account value
        equity: Account equity
        is_trading_blocked: Whether trading is disabled
        is_pattern_day_trader: PDT flag status
    """

    cash: Decimal
    buying_power: Decimal
    portfolio_value: Decimal
    equity: Decimal
    is_trading_blocked: bool = False
    is_pattern_day_trader: bool = False


@dataclass(frozen=True)
class PositionInfo:
    """
    Single position snapshot.

    Attributes:
        symbol: Ticker symbol
        qty: Number of shares/units (positive for long)
        avg_entry_price: Average cost basis per share
        market_value: Current market value
        unrealized_pnl: Unrealized profit/loss
        unrealized_pnl_pct: Unrealized P&L as percentage
        asset_class: Equity or Crypto
        qty_available: Quantity available for trading
    """

    symbol: str
    qty: Decimal
    avg_entry_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal
    asset_class: AssetClassEnum
    qty_available: Decimal


@dataclass(frozen=True)
class OrderRequest:
    """
    Order submission request.

    Attributes:
        symbol: Ticker symbol
        qty: Quantity to trade
        side: BUY or SELL
        order_type: MARKET, LIMIT, or STOP_LIMIT
        limit_price: Limit price (required for LIMIT/STOP_LIMIT)
        stop_price: Stop trigger price (required for STOP_LIMIT)
        client_order_id: Optional client-specified ID for tracking
    """

    symbol: str
    qty: Decimal
    side: OrderSideEnum
    order_type: OrderTypeEnum = OrderTypeEnum.MARKET
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    client_order_id: Optional[str] = None


@dataclass(frozen=True)
class OrderResult:
    """
    Result of order submission or query.

    Attributes:
        order_id: Broker-assigned order ID
        client_order_id: Client-specified ID (if provided)
        symbol: Ticker symbol
        qty: Requested quantity
        filled_qty: Quantity filled so far
        filled_avg_price: Average fill price
        status: Current order status
        created_at: Order creation timestamp
        filled_at: Fill completion timestamp (if filled)
        rejected_reason: Rejection reason (if rejected)
    """

    order_id: str
    client_order_id: Optional[str]
    symbol: str
    qty: Decimal
    filled_qty: Decimal
    filled_avg_price: Optional[Decimal]
    status: OrderStatusEnum
    created_at: datetime
    filled_at: Optional[datetime] = None
    rejected_reason: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in (
            OrderStatusEnum.FILLED,
            OrderStatusEnum.CANCELED,
            OrderStatusEnum.REJECTED,
            OrderStatusEnum.EXPIRED,
        )

    @property
    def is_partial(self) -> bool:
        """Check if order is partially filled."""
        return self.status == OrderStatusEnum.PARTIALLY_FILLED
