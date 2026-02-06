"""
OrderRouter: Portfolio Reconciliation Engine.

Converts strategy signals into executable orders by comparing
target allocations against current positions.

Enterprise Pattern: Target vs Actual Reconciliation
- For each held position: if not in signals or signal ≤ 0, SELL
- For each positive signal: if not held, BUY
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


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of portfolio reconciliation."""

    closes: List[OrderRequest]  # Positions to close (sells)
    opens: List[OrderRequest]  # Positions to open (buys)

    @property
    def total_orders(self) -> int:
        return len(self.closes) + len(self.opens)


class OrderRouter:
    """
    Portfolio Reconciliation Engine.

    Compares strategy signals (target allocations) against current positions
    (actual allocations) and generates orders to reconcile the difference.

    Reconciliation Logic:
    1. CLOSE: For each held position NOT in signals (or signal ≤ 0), sell all
    2. OPEN: For each positive signal without a position, buy to target weight

    Safety Layers (preserved):
    - Dry run mode
    - Max order value limits
    - Position size limits
    - Decimal precision for all monetary values

    Usage:
        router = OrderRouter(executor, config)
        signals = {"AAPL": 0.8, "MSFT": 0.0}  # Want AAPL, exit MSFT
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
        Reconcile portfolio with target signals.

        This method performs a full portfolio reconciliation:
        1. Identifies positions to CLOSE (held but signal ≤ 0 or missing)
        2. Identifies positions to OPEN (signal > 0 but not held)
        3. Submits orders in sequence: closes first, then opens

        Args:
            signals: Dict mapping symbol -> signal strength (0 to 1)
                     - strength > 0: Want to hold this position
                     - strength ≤ 0 or missing: Want to exit this position
            dry_run: If True, calculate orders but don't submit

        Returns:
            List of OrderResult for submitted orders
        """
        # Get current account state
        account = self._executor.get_account_info()
        positions = self._executor.get_positions()

        # Perform reconciliation
        reconciliation = self._reconcile_portfolio(signals, account, positions)

        logger.info(
            f"Reconciliation: {len(reconciliation.closes)} closes, "
            f"{len(reconciliation.opens)} opens"
        )

        if dry_run:
            logger.info(f"DRY RUN: Would submit {reconciliation.total_orders} orders")
            for req in reconciliation.closes:
                logger.info(f"  [CLOSE] SELL {req.qty} {req.symbol}")
            for req in reconciliation.opens:
                logger.info(f"  [OPEN] BUY {req.qty} {req.symbol}")
            return []

        # Submit orders: closes first, then opens
        results: List[OrderResult] = []

        # Execute close orders first (free up capital)
        for request in reconciliation.closes:
            result = self._submit_order(request)
            if result:
                results.append(result)

        # Execute open orders
        for request in reconciliation.opens:
            result = self._submit_order(request)
            if result:
                results.append(result)

        return results

    def _submit_order(self, request: OrderRequest) -> Optional[OrderResult]:
        """Submit a single order with error handling."""
        try:
            result = self._executor.submit_order(request)

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
            return result

        except ExecutorError as e:
            logger.error(f"Failed to submit order for {request.symbol}: {e}")
            return None

    def _reconcile_portfolio(
        self,
        signals: Dict[str, float],
        account: AccountInfo,
        positions: Dict[str, PositionInfo],
    ) -> ReconciliationResult:
        """
        Compare target signals against current positions.

        Returns:
            ReconciliationResult with close and open orders
        """
        close_orders: List[OrderRequest] = []
        open_orders: List[OrderRequest] = []

        # Step 1: Generate CLOSE orders for positions we should exit
        close_orders = self._generate_close_orders(signals, positions)

        # Step 2: Generate OPEN orders for new positions
        open_orders = self._generate_open_orders(signals, account, positions)

        return ReconciliationResult(closes=close_orders, opens=open_orders)

    def _generate_close_orders(
        self,
        signals: Dict[str, float],
        positions: Dict[str, PositionInfo],
    ) -> List[OrderRequest]:
        """
        Generate SELL orders for positions that should be closed.

        A position should be closed if:
        1. Symbol is not in the signals dict, OR
        2. Signal strength is ≤ 0

        Args:
            signals: Target signals from strategy
            positions: Current positions from broker

        Returns:
            List of SELL OrderRequests
        """
        close_orders: List[OrderRequest] = []

        for symbol, position in positions.items():
            signal_strength = signals.get(symbol, 0.0)

            # Close if signal is zero/negative or symbol not in signals
            if signal_strength <= 0:
                # Only close if we have a positive quantity
                if position.qty > 0:
                    logger.info(
                        f"Closing position {symbol}: signal={signal_strength:.2f}, "
                        f"qty={position.qty}"
                    )

                    close_orders.append(
                        OrderRequest(
                            symbol=symbol,
                            qty=position.qty,
                            side=OrderSideEnum.SELL,
                            client_order_id=f"close-{str(uuid.uuid4())[:8]}",
                        )
                    )

        return close_orders

    def _generate_open_orders(
        self,
        signals: Dict[str, float],
        account: AccountInfo,
        positions: Dict[str, PositionInfo],
    ) -> List[OrderRequest]:
        """
        Generate BUY orders for new positions.

        A position should be opened if:
        1. Signal strength > 0, AND
        2. Symbol is NOT in current positions

        Args:
            signals: Target signals from strategy
            account: Current account info
            positions: Current positions from broker

        Returns:
            List of BUY OrderRequests
        """
        # Calculate available capital by asset class
        equity_budget = account.cash * self._config.equity_allocation_pct
        crypto_budget = account.cash * self._config.crypto_allocation_pct

        # Group signals by asset class (only positive signals for new positions)
        equity_signals: Dict[str, float] = {}
        crypto_signals: Dict[str, float] = {}

        for symbol, strength in signals.items():
            # Only consider positive signals
            if strength <= 0:
                continue

            # Skip if already have position (no need to buy more)
            if symbol in positions:
                continue

            asset_class = self._executor.get_asset_class(symbol)
            if asset_class == AssetClassEnum.CRYPTO:
                crypto_signals[symbol] = strength
            else:
                equity_signals[symbol] = strength

        # Convert signals to orders
        orders: List[OrderRequest] = []

        orders.extend(
            self._signals_to_buy_orders(
                equity_signals, equity_budget, account.portfolio_value
            )
        )

        orders.extend(
            self._signals_to_buy_orders(
                crypto_signals, crypto_budget, account.portfolio_value
            )
        )

        return orders

    def _signals_to_buy_orders(
        self,
        signals: Dict[str, float],
        budget: Decimal,
        portfolio_value: Decimal,
    ) -> List[OrderRequest]:
        """
        Convert positive signals to BUY orders with position sizing.

        Args:
            signals: Dict of symbol -> strength (all should be > 0)
            budget: Available budget for this asset class
            portfolio_value: Total portfolio value for position limits

        Returns:
            List of BUY OrderRequests
        """
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
                logger.debug(
                    f"Skipping {symbol}: target value ${target_value:.2f} "
                    f"below minimum ${self._config.min_order_value}"
                )
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
                    client_order_id=f"open-{str(uuid.uuid4())[:8]}",
                )
            )

        return orders

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

    def close_all_positions(self, dry_run: bool = False) -> List[OrderResult]:
        """
        Emergency close of all positions.

        Args:
            dry_run: If True, log but don't execute

        Returns:
            List of OrderResults for close orders
        """
        positions = self._executor.get_positions()

        if not positions:
            logger.info("No positions to close")
            return []

        close_orders = self._generate_close_orders(signals={}, positions=positions)

        if dry_run:
            logger.info(f"DRY RUN: Would close {len(close_orders)} positions")
            for req in close_orders:
                logger.info(f"  [CLOSE] SELL {req.qty} {req.symbol}")
            return []

        results: List[OrderResult] = []
        for request in close_orders:
            result = self._submit_order(request)
            if result:
                results.append(result)

        return results

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol."""
        if self._price_fetcher:
            price = self._price_fetcher(symbol)
            if price and price > 0:
                return price

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
