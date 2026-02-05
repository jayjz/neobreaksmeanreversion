"""
Base strategy class for hybrid reversal trading.
"""
from __future__ import annotations

import backtrader as bt
from abc import abstractmethod
from typing import Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import date

if TYPE_CHECKING:
    from backtrader.order import Order
    from backtrader.trade import Trade


@dataclass
class PositionTracker:
    """Tracks entry information for risk management."""

    entry_price: float
    entry_bar: int
    entry_date: date


class ReversalStrategy(bt.Strategy):
    """
    Abstract base class for mean-reversion strategies.

    Provides:
    - Stop-loss management (5% hard stop)
    - Time-stop management (5 days max hold)
    - Position tracking infrastructure

    Subclasses must implement:
    - generate_signals() -> Dict[str, float]  # ticker -> signal strength
    - get_position_size(data, signal) -> int
    """

    params = (
        ("stop_loss_pct", 0.05),  # 5% hard stop
        ("time_stop_days", 5),  # Exit after 5 days
        ("verbose", False),  # Debug logging
    )

    def __init__(self) -> None:
        """Initialize position tracking dictionaries."""
        self._positions: Dict[str, PositionTracker] = {}
        self._pending_orders: Dict[str, bt.Order] = {}
        self._last_signals: Dict[str, float] = {}  # For signal extraction after run

    def start(self) -> None:
        """Called at strategy start. Override for initialization."""
        pass

    def next(self) -> None:
        """
        Main bar-by-bar logic.

        1. Check risk management (stops) for existing positions
        2. Generate new signals
        3. Execute trades
        """
        # Step 1: Risk management for existing positions
        self._check_risk_management()

        # Step 2: Generate signals (subclass responsibility)
        signals = self.generate_signals()

        # Store signals for extraction after cerebro.run() (Phase 4 integration)
        self._last_signals = signals

        # Step 3: Execute on signals
        for ticker, signal in signals.items():
            if signal != 0:
                self._execute_signal(ticker, signal)

    def _check_risk_management(self) -> None:
        """Check stop-loss and time-stop for all positions."""
        current_bar = len(self.datas[0])

        positions_to_close: list[tuple[str, str]] = []

        for ticker, tracker in self._positions.items():
            try:
                data = self.getdatabyname(ticker)
            except Exception:
                continue

            current_price = data.close[0]

            # Stop-loss check
            loss_pct = (current_price - tracker.entry_price) / tracker.entry_price
            if loss_pct <= -self.p.stop_loss_pct:
                positions_to_close.append((ticker, "stop_loss"))
                continue

            # Time-stop check
            bars_held = current_bar - tracker.entry_bar
            if bars_held >= self.p.time_stop_days:
                positions_to_close.append((ticker, "time_stop"))

        # Close positions
        for ticker, reason in positions_to_close:
            self._close_position(ticker, reason)

    def _close_position(self, ticker: str, reason: str) -> None:
        """Close position for given ticker."""
        try:
            data = self.getdatabyname(ticker)
        except Exception:
            return

        pos = self.getposition(data)

        if pos.size > 0:
            if self.p.verbose:
                print(f"[{self.datas[0].datetime.date(0)}] Closing {ticker}: {reason}")
            self.close(data=data)

        if ticker in self._positions:
            del self._positions[ticker]

    def _execute_signal(self, ticker: str, signal: float) -> None:
        """Execute a trade based on signal."""
        try:
            data = self.getdatabyname(ticker)
        except Exception:
            return

        pos = self.getposition(data)

        # Only enter if no existing position
        if pos.size == 0 and signal > 0:
            size = self.get_position_size(data, signal)
            if size > 0:
                if self.p.verbose:
                    print(
                        f"[{self.datas[0].datetime.date(0)}] BUY {ticker}: size={size}"
                    )
                order = self.buy(data=data, size=size)
                self._pending_orders[ticker] = order

    def notify_order(self, order: "Order") -> None:
        """Track order execution for position management."""
        if order.status == order.Completed:
            # Find which ticker this order belongs to
            ticker = order.data._name

            if order.isbuy():
                # Record entry for risk management
                self._positions[ticker] = PositionTracker(
                    entry_price=order.executed.price,
                    entry_bar=len(self.datas[0]),
                    entry_date=self.datas[0].datetime.date(0),
                )
            elif order.issell():
                # Remove from tracking
                if ticker in self._positions:
                    del self._positions[ticker]

            # Clear pending order
            if ticker in self._pending_orders:
                del self._pending_orders[ticker]

    def notify_trade(self, trade: "Trade") -> None:
        """Log trade completions."""
        if trade.isclosed and self.p.verbose:
            print(f"Trade closed: {trade.data._name}, PnL: {trade.pnl:.2f}")

    @abstractmethod
    def generate_signals(self) -> Dict[str, float]:
        """
        Generate trading signals for all instruments.

        Returns:
            Dict mapping ticker -> signal strength (0 = no signal, >0 = buy strength)
        """
        raise NotImplementedError

    @abstractmethod
    def get_position_size(self, data: bt.AbstractDataBase, signal: float) -> int:
        """
        Calculate position size for a given signal.

        Args:
            data: The data feed for the instrument
            signal: Signal strength (higher = stronger conviction)

        Returns:
            Number of shares/units to trade
        """
        raise NotImplementedError

    @property
    def last_signals(self) -> Dict[str, float]:
        """
        Return signals from the most recent bar.

        Used for signal extraction after cerebro.run() in live/paper trading.
        The returned dict maps ticker -> signal strength.
        """
        return self._last_signals.copy()
