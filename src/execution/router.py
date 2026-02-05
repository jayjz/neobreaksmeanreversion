"""
OrderRouter: Maps strategy signals to executable orders.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Callable

from .base import ExecutorBase, ExecutorError
from .models import (
    OrderRequest,
    OrderResult,
    PositionInfo,
    AccountInfo,
    OrderSideEnum,
    AssetClassEnum,
)

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal processing."""

    max_position_pct: Decimal = Decimal("0.10")  # Max 10% per position
    min_order_value: Decimal = Decimal("100")  # Minimum order $100
    crypto_allocation_pct: Decimal = Decimal("0.30")  # Max 30% crypto
    equity_allocation_pct: Decimal = Decimal("0.70")  # Max 70% equity


@dataclass
class PendingOrder:
    """Tracks order submitted but not yet filled."""

    order_id: str
    symbol: str
    requested_qty: Decimal
    side: OrderSideEnum
    submitted_at: float  # timestamp


class OrderRouter:
    """
    Routes strategy signals to orders via an Executor.

    Responsibilities:
    1. Convert signal strengths (Dict[str, float]) to order quantities
    2. Apply position sizing rules
    3. Track pending orders
    4. Handle partial fills
    5. Provide portfolio synchronization

    Usage:
        router = OrderRouter(executor, config)

        # From strategy's generate_signals()
        signals = {"AAPL": 0.8, "MSFT": 0.5, "BTC-USD": 0.6}

        # Execute signals
        orders = router.execute_signals(signals)
    """

    def __init__(
        self,
        executor: ExecutorBase,
        config: Optional[SignalConfig] = None,
        price_fetcher: Optional[Callable[[str], Decimal]] = None,
    ) -> None:
        """
        Initialize OrderRouter.

        Args:
            executor: ExecutorBase implementation for order execution
            config: Position sizing and allocation configuration
            price_fetcher: Optional function to get current price for a symbol.
                           If None, uses position's current_price or estimates.
        """
        self._executor = executor
        self._config = config or SignalConfig()
        self._price_fetcher = price_fetcher

        # Track pending orders
        self._pending_orders: Dict[str, PendingOrder] = {}

    def execute_signals(
        self,
        signals: Dict[str, float],
        dry_run: bool = False,
    ) -> List[OrderResult]:
        """
        Convert signals to orders and execute.

        Args:
            signals: Dict mapping symbol -> signal strength (0 to 1)
                     Higher strength = stronger conviction
            dry_run: If True, calculate orders but don't submit

        Returns:
            List of OrderResult for submitted orders
        """
        if not signals:
            return []

        # Get current account state
        account = self._executor.get_account_info()
        positions = self._executor.get_positions()

        # Calculate target orders
        orders_to_submit = self._calculate_orders(signals, account, positions)

        if dry_run:
            logger.info(f"DRY RUN: Would submit {len(orders_to_submit)} orders")
            for req in orders_to_submit:
                logger.info(f"  {req.side.name} {req.qty} {req.symbol}")
            return []

        # Submit orders
        results: List[OrderResult] = []
        for request in orders_to_submit:
            try:
                result = self._executor.submit_order(request)
                results.append(result)

                # Track pending order
                self._pending_orders[result.order_id] = PendingOrder(
                    order_id=result.order_id,
                    symbol=request.symbol,
                    requested_qty=request.qty,
                    side=request.side,
                    submitted_at=result.created_at.timestamp(),
                )

                logger.info(
                    f"Submitted {request.side.name} {request.qty} {request.symbol}"
                )
            except ExecutorError as e:
                logger.error(f"Failed to submit order for {request.symbol}: {e}")

        return results

    def sync_portfolio(self) -> Dict[str, PositionInfo]:
        """
        Synchronize internal state with broker positions.

        Returns:
            Current positions from broker
        """
        positions = self._executor.get_positions()

        # Update pending orders based on current state
        self._update_pending_orders()

        return positions

    def get_pending_orders(self) -> List[PendingOrder]:
        """Get list of orders that haven't reached terminal state."""
        self._update_pending_orders()
        return list(self._pending_orders.values())

    def cancel_pending_orders(self, symbols: Optional[List[str]] = None) -> int:
        """
        Cancel pending orders.

        Args:
            symbols: If provided, only cancel orders for these symbols.
                     If None, cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        orders_to_cancel = list(self._pending_orders.values())

        for pending in orders_to_cancel:
            if symbols and pending.symbol not in symbols:
                continue

            try:
                self._executor.cancel_order(pending.order_id)
                del self._pending_orders[pending.order_id]
                cancelled += 1
            except ExecutorError as e:
                logger.error(f"Failed to cancel order {pending.order_id}: {e}")

        return cancelled

    def handle_partial_fill(self, order_id: str) -> Optional[OrderResult]:
        """
        Check and handle partial fills.

        This method returns the current order state for decision making.

        Args:
            order_id: Order ID to check

        Returns:
            Current OrderResult, or None if order not found
        """
        try:
            result = self._executor.get_order(order_id)

            if result.is_partial:
                logger.info(
                    f"Order {order_id} partially filled: "
                    f"{result.filled_qty}/{result.qty}"
                )

            # Remove from pending if terminal
            if result.is_terminal and order_id in self._pending_orders:
                del self._pending_orders[order_id]

            return result
        except ExecutorError:
            return None

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _calculate_orders(
        self,
        signals: Dict[str, float],
        account: AccountInfo,
        positions: Dict[str, PositionInfo],
    ) -> List[OrderRequest]:
        """
        Convert signals to order requests with position sizing.

        Algorithm:
        1. Filter signals to only those without existing positions
        2. Separate equity vs crypto signals
        3. Apply allocation limits
        4. Calculate per-symbol allocation
        5. Convert to share quantities
        """
        orders: List[OrderRequest] = []

        # Calculate available capital by asset class
        equity_budget = account.cash * self._config.equity_allocation_pct
        crypto_budget = account.cash * self._config.crypto_allocation_pct

        # Group signals by asset class
        equity_signals: Dict[str, float] = {}
        crypto_signals: Dict[str, float] = {}

        for symbol, strength in signals.items():
            if strength <= 0:
                continue

            # Skip if already have position
            if symbol in positions:
                continue

            asset_class = self._executor.get_asset_class(symbol)
            if asset_class == AssetClassEnum.CRYPTO:
                crypto_signals[symbol] = strength
            else:
                equity_signals[symbol] = strength

        # Process equity signals
        orders.extend(
            self._signals_to_orders(
                equity_signals, equity_budget, account.portfolio_value
            )
        )

        # Process crypto signals
        orders.extend(
            self._signals_to_orders(
                crypto_signals, crypto_budget, account.portfolio_value
            )
        )

        return orders

    def _signals_to_orders(
        self,
        signals: Dict[str, float],
        budget: Decimal,
        portfolio_value: Decimal,
    ) -> List[OrderRequest]:
        """Convert signals within an asset class to orders."""
        if not signals:
            return []

        orders: List[OrderRequest] = []

        # Normalize signal strengths to allocations
        total_strength = sum(signals.values())

        for symbol, strength in signals.items():
            # Proportional allocation based on signal strength
            allocation_pct = Decimal(str(strength / total_strength))
            target_value = budget * allocation_pct

            # Cap at max position size
            max_value = portfolio_value * self._config.max_position_pct
            target_value = min(target_value, max_value)

            # Skip if below minimum
            if target_value < self._config.min_order_value:
                continue

            # Get current price to calculate quantity
            price = self._get_price(symbol)
            if price is None or price <= 0:
                logger.warning(f"Cannot get price for {symbol}, skipping")
                continue

            qty = target_value / price

            # Round appropriately (crypto can be fractional, equity integers)
            asset_class = self._executor.get_asset_class(symbol)
            if asset_class == AssetClassEnum.EQUITY:
                qty = Decimal(int(qty))  # Round down to whole shares
            else:
                qty = qty.quantize(Decimal("0.0001"))  # 4 decimal places for crypto

            if qty <= 0:
                continue

            orders.append(
                OrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSideEnum.BUY,
                    client_order_id=str(uuid.uuid4())[:8],
                )
            )

        return orders

    def _get_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol."""
        if self._price_fetcher:
            return self._price_fetcher(symbol)

        # Fallback: try to get from existing position
        try:
            pos = self._executor.get_position(symbol)
            if pos:
                return pos.avg_entry_price
        except ExecutorError:
            pass

        return None

    def _update_pending_orders(self) -> None:
        """Update status of pending orders."""
        for order_id in list(self._pending_orders.keys()):
            try:
                result = self._executor.get_order(order_id)
                if result.is_terminal:
                    del self._pending_orders[order_id]
            except ExecutorError:
                # Order may have been purged
                del self._pending_orders[order_id]
