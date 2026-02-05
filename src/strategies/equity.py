"""
Equity mean-reversion strategy.
Buys bottom decile of weekly performers on Friday rebalance.
"""
from __future__ import annotations

import backtrader as bt
from typing import Dict, List, Tuple

from .base import ReversalStrategy


class EquityReversal(ReversalStrategy):
    """
    S&P 500 Mean Reversion Strategy.

    Logic:
    - Rebalance weekly (Friday close)
    - Calculate 1-week (5-day) returns for all equity feeds
    - Long bottom decile (worst performers expecting reversion)
    - Equal weight allocation among selected

    Inherits risk management from ReversalStrategy:
    - 5% hard stop-loss
    - 5-day time stop
    """

    params = (  # type: ignore[assignment]
        ("rebalance_weekday", 4),  # Friday = 4
        ("lookback_days", 5),  # 1 week lookback
        ("decile_pct", 0.10),  # Bottom 10%
        ("max_positions", 10),  # Max concurrent positions
        # Inherited: stop_loss_pct, time_stop_days, verbose
    )

    def __init__(self) -> None:
        super().__init__()

        # Store references to equity data feeds
        self._equity_datas: List[bt.AbstractDataBase] = []

    def start(self) -> None:
        """Initialize data feed categorization."""
        super().start()

        # Identify equity data feeds (all feeds for this strategy)
        self._equity_datas = list(self.datas)

        if self.p.verbose:
            print(
                f"EquityReversal initialized with {len(self._equity_datas)} equities"
            )

    def generate_signals(self) -> Dict[str, float]:
        """
        Generate buy signals for bottom decile performers on Friday.

        Returns:
            Dict[ticker, signal_strength] where signal > 0 means buy
        """
        signals: Dict[str, float] = {}

        # Only rebalance on specified weekday
        current_date = self.datas[0].datetime.date(0)
        if current_date.weekday() != self.p.rebalance_weekday:
            return signals

        # Calculate 5-day returns for all equities
        returns = self._calculate_returns()

        if not returns:
            return signals

        # Rank and select bottom decile
        selected = self._select_bottom_decile(returns)

        # Generate equal-weight signals for selected
        signal_strength = 1.0 / max(1, len(selected))
        for ticker in selected:
            # Only signal if not already in position
            try:
                data = self.getdatabyname(ticker)
                if self.getposition(data).size == 0:
                    signals[ticker] = signal_strength
            except Exception:
                continue

        return signals

    def _calculate_returns(self) -> Dict[str, float]:
        """Calculate lookback returns for all equity feeds."""
        returns: Dict[str, float] = {}

        for data in self._equity_datas:
            # Ensure enough history
            if len(data) < self.p.lookback_days + 1:
                continue

            # Calculate return
            current_close = data.close[0]
            past_close = data.close[-self.p.lookback_days]

            if past_close > 0:
                ret = (current_close - past_close) / past_close
                returns[data._name] = ret

        return returns

    def _select_bottom_decile(self, returns: Dict[str, float]) -> List[str]:
        """Select bottom decile of performers (worst returns)."""
        if not returns:
            return []

        # Sort by return (ascending - worst first)
        sorted_returns: List[Tuple[str, float]] = sorted(
            returns.items(), key=lambda x: x[1]
        )

        # Calculate decile size
        decile_size = max(1, int(len(sorted_returns) * self.p.decile_pct))
        decile_size = min(decile_size, self.p.max_positions)

        # Select bottom performers
        selected = [ticker for ticker, _ in sorted_returns[:decile_size]]

        if self.p.verbose:
            print(
                f"[{self.datas[0].datetime.date(0)}] Selected {len(selected)} equities:"
            )
            for ticker, ret in sorted_returns[:decile_size]:
                print(f"  {ticker}: {ret * 100:.2f}%")

        return selected

    def get_position_size(self, data: bt.AbstractDataBase, signal: float) -> int:
        """
        Calculate equal-weight position size.

        Allocates portfolio value equally among all signals.
        """
        # Get available cash
        cash = self.broker.getcash()

        # Equal weight based on signal (signal encodes 1/N)
        allocation = cash * signal * 0.95  # Keep 5% buffer

        # Calculate shares
        current_price = data.close[0]
        if current_price <= 0:
            return 0

        shares = int(allocation / current_price)
        return max(0, shares)
