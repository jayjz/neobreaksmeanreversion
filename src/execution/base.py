"""
Abstract base class for execution engines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .models import (
    AccountInfo,
    PositionInfo,
    OrderRequest,
    OrderResult,
    AssetClassEnum,
)


class ExecutorError(Exception):
    """Base exception for executor errors."""

    pass


class SafetyError(ExecutorError):
    """Raised when safety checks prevent an operation."""

    pass


class ExecutorBase(ABC):
    """
    Abstract interface for trade execution.

    All broker implementations must inherit from this class.
    Provides a consistent API for account management, position tracking,
    and order management.
    """

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Retrieve current account information.

        Returns:
            AccountInfo with cash, buying power, portfolio value, etc.

        Raises:
            ExecutorError: If account cannot be retrieved
        """
        ...

    @abstractmethod
    def get_positions(self) -> Dict[str, PositionInfo]:
        """
        Retrieve all open positions.

        Returns:
            Dict mapping symbol -> PositionInfo

        Raises:
            ExecutorError: If positions cannot be retrieved
        """
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """
        Retrieve position for specific symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            PositionInfo if position exists, None otherwise

        Raises:
            ExecutorError: If position cannot be retrieved
        """
        ...

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit a new order.

        Args:
            request: OrderRequest with symbol, qty, side, type

        Returns:
            OrderResult with order status and details

        Raises:
            ExecutorError: If order submission fails
            SafetyError: If safety checks prevent order
        """
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> OrderResult:
        """
        Retrieve order status by ID.

        Args:
            order_id: Broker-assigned order ID

        Returns:
            OrderResult with current status

        Raises:
            ExecutorError: If order not found
        """
        ...

    @abstractmethod
    def get_open_orders(self) -> List[OrderResult]:
        """
        Retrieve all open (non-terminal) orders.

        Returns:
            List of OrderResult for open orders
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Broker-assigned order ID

        Returns:
            True if cancellation succeeded

        Raises:
            ExecutorError: If cancellation fails
        """
        ...

    @abstractmethod
    def close_position(self, symbol: str) -> Optional[OrderResult]:
        """
        Close entire position for symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            OrderResult for closing order, None if no position

        Raises:
            ExecutorError: If close fails
        """
        ...

    @abstractmethod
    def close_all_positions(self) -> List[OrderResult]:
        """
        Close all open positions.

        Returns:
            List of OrderResult for each closing order

        Raises:
            ExecutorError: If close fails
        """
        ...

    def get_asset_class(self, symbol: str) -> AssetClassEnum:
        """
        Determine asset class from symbol.

        Default implementation uses simple heuristics.
        Override for broker-specific behavior.

        Args:
            symbol: Ticker symbol

        Returns:
            AssetClassEnum.CRYPTO if symbol contains crypto identifiers,
            AssetClassEnum.EQUITY otherwise
        """
        crypto_suffixes = ("-USD", "/USD", "USDT", "USDC")
        crypto_prefixes = ("BTC", "ETH", "SOL", "AVAX", "DOGE")

        symbol_upper = symbol.upper()

        if any(symbol_upper.endswith(s) for s in crypto_suffixes):
            return AssetClassEnum.CRYPTO
        if any(symbol_upper.startswith(p) for p in crypto_prefixes):
            return AssetClassEnum.CRYPTO

        return AssetClassEnum.EQUITY
