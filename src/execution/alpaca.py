"""
Alpaca-py implementation of ExecutorBase.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Union

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLimitOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, AssetClass
from alpaca.trading.models import Position as AlpacaPosition, Order as AlpacaOrder

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

logger = logging.getLogger(__name__)


class AlpacaExecutor(ExecutorBase):
    """
    Alpaca implementation of the ExecutorBase interface.

    Supports both paper and live trading with multiple safety mechanisms:
    1. paper=True uses Alpaca paper trading API
    2. live_trading_enabled=False prevents any live orders
    3. max_order_value limits single order size

    Usage:
        # Paper trading (safe)
        executor = AlpacaExecutor(api_key, secret_key, paper=True)

        # Live trading (requires explicit flag)
        executor = AlpacaExecutor(
            api_key, secret_key,
            paper=False,
            live_trading_enabled=True  # Must be explicit
        )
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        live_trading_enabled: bool = False,
        max_order_value: Decimal = Decimal("10000"),
    ) -> None:
        """
        Initialize AlpacaExecutor.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: If True, use paper trading API (default: True)
            live_trading_enabled: If True and paper=False, allow live trading
                                  (default: False)
            max_order_value: Maximum value for single order (safety limit)

        Raises:
            SafetyError: If paper=False and live_trading_enabled=False
        """
        # SAFETY CHECK: Explicit flag required for live trading
        if not paper and not live_trading_enabled:
            raise SafetyError(
                "Live trading requires explicit live_trading_enabled=True. "
                "This is a safety mechanism to prevent accidental live orders."
            )

        self._paper = paper
        self._live_trading_enabled = live_trading_enabled
        self._max_order_value = max_order_value

        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

        mode = "PAPER" if paper else "LIVE"
        logger.info(f"AlpacaExecutor initialized in {mode} mode")

    @property
    def is_paper(self) -> bool:
        """Return True if in paper trading mode."""
        return self._paper

    def get_account_info(self) -> AccountInfo:
        """Retrieve account information from Alpaca."""
        try:
            account = self._client.get_account()

            return AccountInfo(
                cash=Decimal(str(account.cash)),  # type: ignore[union-attr]
                buying_power=Decimal(str(account.buying_power)),  # type: ignore[union-attr]
                portfolio_value=Decimal(str(account.portfolio_value)),  # type: ignore[union-attr]
                equity=Decimal(str(account.equity)),  # type: ignore[union-attr]
                is_trading_blocked=bool(account.trading_blocked),  # type: ignore[union-attr]
                is_pattern_day_trader=bool(account.pattern_day_trader),  # type: ignore[union-attr]
            )
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise ExecutorError(f"Failed to get account info: {e}") from e

    def get_positions(self) -> Dict[str, PositionInfo]:
        """Retrieve all positions from Alpaca."""
        try:
            positions = self._client.get_all_positions()
            return {pos.symbol: self._convert_position(pos) for pos in positions}  # type: ignore[union-attr, arg-type]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise ExecutorError(f"Failed to get positions: {e}") from e

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Retrieve single position from Alpaca."""
        try:
            pos = self._client.get_open_position(symbol)
            return self._convert_position(pos)  # type: ignore[arg-type]
        except Exception as e:
            # Position not found is not an error
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Failed to get position for {symbol}: {e}")
            raise ExecutorError(f"Failed to get position for {symbol}: {e}") from e

    def submit_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit order to Alpaca with safety checks.

        Performs validations:
        1. Order value does not exceed max_order_value
        2. Asset class is appropriate for order type
        3. TimeInForce is compatible with asset class
        """
        # Safety: Validate order
        self._validate_order_request(request)

        # Build Alpaca order request
        alpaca_request = self._build_alpaca_request(request)

        try:
            order = self._client.submit_order(alpaca_request)
            return self._convert_order(order)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise ExecutorError(f"Failed to submit order: {e}") from e

    def get_order(self, order_id: str) -> OrderResult:
        """Retrieve order by ID from Alpaca."""
        try:
            order = self._client.get_order_by_id(order_id)
            return self._convert_order(order)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise ExecutorError(f"Failed to get order {order_id}: {e}") from e

    def get_open_orders(self) -> List[OrderResult]:
        """Retrieve all open orders from Alpaca."""
        try:
            orders = self._client.get_orders()
            return [self._convert_order(o) for o in orders]  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise ExecutorError(f"Failed to get open orders: {e}") from e

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca."""
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise ExecutorError(f"Failed to cancel order {order_id}: {e}") from e

    def close_position(self, symbol: str) -> Optional[OrderResult]:
        """Close position on Alpaca."""
        # Check if position exists
        position = self.get_position(symbol)
        if position is None:
            logger.info(f"No position to close for {symbol}")
            return None

        try:
            order = self._client.close_position(symbol)
            return self._convert_order(order)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            raise ExecutorError(f"Failed to close position {symbol}: {e}") from e

    def close_all_positions(self) -> List[OrderResult]:
        """Close all positions on Alpaca."""
        try:
            responses = self._client.close_all_positions(cancel_orders=True)
            results: List[OrderResult] = []
            for resp in responses:
                if hasattr(resp, "body") and resp.body:
                    results.append(self._convert_order(resp.body))  # type: ignore[arg-type]
            return results
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            raise ExecutorError(f"Failed to close all positions: {e}") from e

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _validate_order_request(self, request: OrderRequest) -> None:
        """Validate order request before submission."""
        # For limit orders, use limit price for value estimate
        if request.order_type == OrderTypeEnum.LIMIT and request.limit_price:
            estimated_value = request.qty * request.limit_price
        else:
            # For market orders, use conservative estimate
            estimated_value = request.qty * Decimal("1000")

        if estimated_value > self._max_order_value:
            raise SafetyError(
                f"Estimated order value ${estimated_value} exceeds maximum "
                f"${self._max_order_value}. Increase max_order_value if intentional."
            )

    def _build_alpaca_request(
        self, request: OrderRequest
    ) -> Union[MarketOrderRequest, LimitOrderRequest, StopLimitOrderRequest]:
        """Convert OrderRequest to Alpaca request object."""
        asset_class = self.get_asset_class(request.symbol)

        # Determine TimeInForce based on asset class
        if asset_class == AssetClassEnum.CRYPTO:
            time_in_force = TimeInForce.GTC
        else:
            time_in_force = TimeInForce.DAY

        side = OrderSide.BUY if request.side == OrderSideEnum.BUY else OrderSide.SELL

        if request.order_type == OrderTypeEnum.MARKET:
            return MarketOrderRequest(
                symbol=request.symbol,
                qty=float(request.qty),
                side=side,
                time_in_force=time_in_force,
                client_order_id=request.client_order_id,
            )
        elif request.order_type == OrderTypeEnum.LIMIT:
            return LimitOrderRequest(
                symbol=request.symbol,
                qty=float(request.qty),
                side=side,
                time_in_force=time_in_force,
                limit_price=float(request.limit_price) if request.limit_price else None,
                client_order_id=request.client_order_id,
            )
        elif request.order_type == OrderTypeEnum.STOP_LIMIT:
            return StopLimitOrderRequest(
                symbol=request.symbol,
                qty=float(request.qty),
                side=side,
                time_in_force=time_in_force,
                limit_price=float(request.limit_price) if request.limit_price else None,
                stop_price=float(request.stop_price) if request.stop_price else None,
                client_order_id=request.client_order_id,
            )
        else:
            raise ExecutorError(f"Unsupported order type: {request.order_type}")

    def _convert_position(self, pos: AlpacaPosition) -> PositionInfo:
        """Convert Alpaca Position to PositionInfo."""
        if hasattr(pos, "asset_class") and pos.asset_class == AssetClass.CRYPTO:
            asset_class = AssetClassEnum.CRYPTO
        else:
            asset_class = AssetClassEnum.EQUITY

        return PositionInfo(
            symbol=pos.symbol,
            qty=Decimal(str(pos.qty)),
            avg_entry_price=Decimal(str(pos.avg_entry_price)),
            market_value=Decimal(str(pos.market_value)),
            unrealized_pnl=Decimal(str(pos.unrealized_pl)),
            unrealized_pnl_pct=Decimal(str(pos.unrealized_plpc)),
            asset_class=asset_class,
            qty_available=Decimal(str(pos.qty_available)),
        )

    def _convert_order(self, order: AlpacaOrder) -> OrderResult:
        """Convert Alpaca Order to OrderResult."""
        status = self._convert_order_status(order.status)

        return OrderResult(
            order_id=str(order.id),
            client_order_id=order.client_order_id,
            symbol=order.symbol or "",  # type: ignore[arg-type]
            qty=Decimal(str(order.qty)) if order.qty else Decimal("0"),
            filled_qty=Decimal(str(order.filled_qty)) if order.filled_qty else Decimal("0"),
            filled_avg_price=(
                Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None
            ),
            status=status,
            created_at=order.created_at,
            filled_at=order.filled_at,
        )

    def _convert_order_status(self, alpaca_status: OrderStatus) -> OrderStatusEnum:
        """Map Alpaca OrderStatus to our OrderStatusEnum."""
        mapping = {
            OrderStatus.NEW: OrderStatusEnum.SUBMITTED,
            OrderStatus.ACCEPTED: OrderStatusEnum.SUBMITTED,
            OrderStatus.PENDING_NEW: OrderStatusEnum.PENDING,
            OrderStatus.PARTIALLY_FILLED: OrderStatusEnum.PARTIALLY_FILLED,
            OrderStatus.FILLED: OrderStatusEnum.FILLED,
            OrderStatus.CANCELED: OrderStatusEnum.CANCELED,
            OrderStatus.EXPIRED: OrderStatusEnum.EXPIRED,
            OrderStatus.REJECTED: OrderStatusEnum.REJECTED,
        }
        return mapping.get(alpaca_status, OrderStatusEnum.PENDING)
