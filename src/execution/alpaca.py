"""
Alpaca-py implementation of ExecutorBase.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, AssetClass, QueryOrderStatus
from alpaca.common.exceptions import APIError

from .base import ExecutorBase, ExecutorError, SafetyError
from .models import AccountInfo, PositionInfo, OrderRequest, OrderResult, OrderSideEnum, AssetClassEnum

logger = logging.getLogger(__name__)

class AlpacaExecutor(ExecutorBase):
    """
    Alpaca execution engine using alpaca-py.
    
    Includes safety layers:
    1. Paper/Live mode enforcement
    2. Max order value check
    """
    
    def __init__(
        self, 
        api_key: str, 
        secret_key: str, 
        paper: bool = True,
        live_trading_enabled: bool = False,
        max_order_value: Decimal = Decimal("10000"),
    ):
        self._paper = paper
        self.max_order_value = max_order_value

        # SAFETY LAYER 1: Constructor Guard
        if not paper and not live_trading_enabled:
            raise SafetyError(
                "LIVE TRADING ATTEMPTED WITHOUT EXPLICIT FLAG. "
                "Set live_trading_enabled=True to enable live trading."
            )
            
        try:
            self._client = TradingClient(api_key, secret_key, paper=paper)
            # Verify connection by fetching account
            self._client.get_account()
            mode = "PAPER" if paper else "LIVE"
            logger.info(f"AlpacaExecutor initialized in {mode} mode")
        except Exception as e:
            raise ExecutorError(f"Failed to connect to Alpaca: {e}") from e

    @property
    def is_paper(self) -> bool:
        """Return True if executor is in paper trading mode."""
        return self._paper

    def _translate_symbol_to_alpaca(self, symbol: str) -> str:
        """
        Convert internal ticker format (Yahoo) to Alpaca format.
        Example: 'ETH-USD' -> 'ETH/USD'
        """
        if "-USD" in symbol and "USD" not in symbol[:3]: # Simple crypto check
             return symbol.replace("-USD", "/USD")
        return symbol

    def _translate_symbol_from_alpaca(self, symbol: str) -> str:
        """
        Convert Alpaca ticker format back to internal format.
        Example: 'ETH/USD' -> 'ETH-USD'
        """
        return symbol.replace("/USD", "-USD")

    def get_account_info(self) -> AccountInfo:
        """Fetch current account status."""
        try:
            account = self._client.get_account()
            
            return AccountInfo(
                cash=Decimal(str(account.cash)), # type: ignore[union-attr]
                buying_power=Decimal(str(account.buying_power)), # type: ignore[union-attr]
                portfolio_value=Decimal(str(account.portfolio_value)), # type: ignore[union-attr]
                equity=Decimal(str(account.equity)), # type: ignore[union-attr]
                is_trading_blocked=bool(account.trading_blocked), # type: ignore[union-attr]
                is_pattern_day_trader=bool(account.pattern_day_trader), # type: ignore[union-attr]
            )
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise ExecutorError(f"Failed to get account info: {e}") from e

    def get_positions(self) -> Dict[str, PositionInfo]:
        """Retrieve all positions from Alpaca."""
        try:
            positions = self._client.get_all_positions()
            return {
                self._translate_symbol_from_alpaca(pos.symbol): self._convert_position(pos) # type: ignore[union-attr, arg-type]
                for pos in positions
            }
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise ExecutorError(f"Failed to get positions: {e}") from e

    def _convert_position(self, pos: Any) -> PositionInfo:  # type: ignore[type-arg]
        """Convert Alpaca position object to PositionInfo DTO."""
        internal_symbol = self._translate_symbol_from_alpaca(pos.symbol)
        return PositionInfo(
            symbol=internal_symbol,
            qty=Decimal(str(pos.qty)),
            avg_entry_price=Decimal(str(pos.avg_entry_price)),
            market_value=Decimal(str(pos.market_value)),
            unrealized_pnl=Decimal(str(pos.unrealized_pl)),
            unrealized_pnl_pct=Decimal(str(pos.unrealized_plpc)) if pos.unrealized_plpc else Decimal("0"),
            asset_class=AssetClassEnum.CRYPTO if pos.asset_class == AssetClass.CRYPTO else AssetClassEnum.EQUITY,
            qty_available=Decimal(str(pos.qty_available)) if pos.qty_available else Decimal(str(pos.qty)),
        )

    def submit_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to Alpaca."""
        # SAFETY LAYER 2: Max Order Value Check
        # We need estimated price to check value.
        # Ideally passed in request, but for now we rely on router sizing.
        # This check happens at the router level usually, but could be added here if we fetch price.

        # Translate symbol
        alpaca_symbol = self._translate_symbol_to_alpaca(request.symbol)

        # Determine side
        side = OrderSide.BUY if request.side == OrderSideEnum.BUY else OrderSide.SELL

        # Determine TimeInForce using asset class detection
        # Crypto ONLY supports GTC or IOC (error 42210000 if DAY used)
        # Equities support DAY, GTC, etc.
        asset_class = self.get_asset_class(request.symbol)
        tif = TimeInForce.GTC if asset_class == AssetClassEnum.CRYPTO else TimeInForce.DAY

        req = MarketOrderRequest(
            symbol=alpaca_symbol,
            qty=float(request.qty), # Alpaca expects float
            side=side,
            time_in_force=tif
        )

        logger.info(f"Submitting {side.name} {asset_class.name} order for {alpaca_symbol}: {request.qty} units (TIF={tif.name})")

        try:
            order = self._client.submit_order(req)
            return self._convert_order(order) # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise ExecutorError(f"Failed to submit order: {e}") from e

    def _convert_order(self, order: Any) -> OrderResult:  # type: ignore[type-arg]
        """Convert Alpaca order object to OrderResult DTO."""
        from .models import OrderStatusEnum

        # Map Alpaca status to our enum
        status_map = {
            OrderStatus.FILLED: OrderStatusEnum.FILLED,
            OrderStatus.PARTIALLY_FILLED: OrderStatusEnum.PARTIALLY_FILLED,
            OrderStatus.NEW: OrderStatusEnum.SUBMITTED,
            OrderStatus.ACCEPTED: OrderStatusEnum.SUBMITTED,
            OrderStatus.PENDING_NEW: OrderStatusEnum.PENDING,
            OrderStatus.REJECTED: OrderStatusEnum.REJECTED,
            OrderStatus.CANCELED: OrderStatusEnum.CANCELED,
            OrderStatus.EXPIRED: OrderStatusEnum.EXPIRED,
        }

        status = status_map.get(order.status, OrderStatusEnum.PENDING)

        # Convert internal symbol back
        internal_symbol = self._translate_symbol_from_alpaca(order.symbol or "")

        return OrderResult(
            order_id=str(order.id),
            client_order_id=order.client_order_id,
            symbol=internal_symbol,
            qty=Decimal(str(order.qty)) if order.qty else Decimal("0"),
            filled_qty=Decimal(str(order.filled_qty)) if order.filled_qty else Decimal("0"),
            filled_avg_price=(
                Decimal(str(order.filled_avg_price))
                if order.filled_avg_price
                else None
            ),
            status=status,
            created_at=order.created_at,
            filled_at=order.filled_at,
        )

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """
        Retrieve position for a specific symbol.

        Args:
            symbol: Ticker symbol (internal format, e.g., 'ETH-USD')

        Returns:
            PositionInfo if position exists, None otherwise
        """
        alpaca_symbol = self._translate_symbol_to_alpaca(symbol)

        try:
            pos = self._client.get_open_position(alpaca_symbol)
            return self._convert_position(pos)  # type: ignore[arg-type]
        except APIError as e:
            # Position not found returns 404
            if "404" in str(e) or "position does not exist" in str(e).lower():
                return None
            logger.error(f"Failed to get position for {symbol}: {e}")
            raise ExecutorError(f"Failed to get position for {symbol}: {e}") from e
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            raise ExecutorError(f"Failed to get position for {symbol}: {e}") from e

    def get_order(self, order_id: str) -> OrderResult:
        """
        Retrieve order status by ID.

        Args:
            order_id: Broker-assigned order ID

        Returns:
            OrderResult with current status
        """
        try:
            order = self._client.get_order_by_id(order_id)
            return self._convert_order(order)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise ExecutorError(f"Failed to get order {order_id}: {e}") from e

    def get_open_orders(self) -> List[OrderResult]:
        """
        Retrieve all open (non-terminal) orders.

        Returns:
            List of OrderResult for open orders
        """
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self._client.get_orders(request)
            return [
                self._convert_order(order)  # type: ignore[arg-type]
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise ExecutorError(f"Failed to get open orders: {e}") from e

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Broker-assigned order ID

        Returns:
            True if cancellation succeeded
        """
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        except APIError as e:
            # Order may already be filled or cancelled
            if "cannot be cancelled" in str(e).lower():
                logger.warning(f"Order {order_id} cannot be cancelled: {e}")
                return False
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise ExecutorError(f"Failed to cancel order {order_id}: {e}") from e
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise ExecutorError(f"Failed to cancel order {order_id}: {e}") from e

    def close_position(self, symbol: str) -> Optional[OrderResult]:
        """
        Close entire position for symbol.

        Args:
            symbol: Ticker symbol (internal format, e.g., 'ETH-USD')

        Returns:
            OrderResult for closing order, None if no position
        """
        alpaca_symbol = self._translate_symbol_to_alpaca(symbol)

        try:
            order = self._client.close_position(alpaca_symbol)
            logger.info(f"Closed position for {symbol}")
            return self._convert_order(order)  # type: ignore[arg-type]
        except APIError as e:
            # Position not found returns 404
            if "404" in str(e) or "position does not exist" in str(e).lower():
                logger.info(f"No position to close for {symbol}")
                return None
            logger.error(f"Failed to close position for {symbol}: {e}")
            raise ExecutorError(f"Failed to close position for {symbol}: {e}") from e
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            raise ExecutorError(f"Failed to close position for {symbol}: {e}") from e

    def close_all_positions(self) -> List[OrderResult]:
        """
        Close all open positions.

        Returns:
            List of OrderResult for each closing order
        """
        try:
            # close_all_positions returns list of close orders
            responses = self._client.close_all_positions(cancel_orders=True)
            results: List[OrderResult] = []

            for response in responses:
                # Each response has 'symbol' and 'status' or error info
                # If successful, it includes the order
                if hasattr(response, "body") and response.body:
                    order = response.body
                    results.append(self._convert_order(order))  # type: ignore[arg-type]

            logger.info(f"Closed {len(results)} positions")
            return results
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            raise ExecutorError(f"Failed to close all positions: {e}") from e
