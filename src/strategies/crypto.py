"""
Crypto volatility-gated mean-reversion strategy.
Only trades when market is ranging (ADX < 25).
"""
from __future__ import annotations

import backtrader as bt
from typing import Dict, Optional

from .base import ReversalStrategy


class CryptoReversal(ReversalStrategy):
    """
    Crypto Mean Reversion Strategy with ADX Regime Filter.

    Logic:
    - Calculate ADX for each crypto asset
    - Only consider trades when ADX < 25 (ranging market)
    - In ranging regime: buy on dips (negative short-term return)
    - Apply same risk management as equities

    ADX Interpretation:
    - ADX < 25: Weak trend / ranging -> Good for mean reversion
    - ADX > 25: Strong trend -> Avoid mean reversion
    """

    params = (  # type: ignore[assignment]
        ("adx_period", 14),  # ADX calculation period
        ("adx_threshold", 25.0),  # Max ADX for trading
        ("lookback_days", 3),  # Short lookback for crypto
        ("dip_threshold", -0.03),  # 3% dip triggers buy
        ("max_positions", 3),  # Max concurrent crypto positions
        # Inherited: stop_loss_pct, time_stop_days, verbose
    )

    def __init__(self) -> None:
        super().__init__()

        # Store ADX indicators per data feed
        self._adx_indicators: Dict[str, bt.indicators.AverageDirectionalMovementIndex] = {}

        # Create ADX indicator for each data feed
        for data in self.datas:
            self._adx_indicators[data._name] = bt.indicators.AverageDirectionalMovementIndex(
                data,
                period=self.p.adx_period,
            )

    def start(self) -> None:
        """Log initialization."""
        super().start()
        if self.p.verbose:
            print(f"CryptoReversal initialized with {len(self.datas)} crypto assets")
            print(f"ADX threshold: {self.p.adx_threshold}")

    def generate_signals(self) -> Dict[str, float]:
        """
        Generate buy signals for ranging crypto markets on dips.

        Returns:
            Dict[ticker, signal_strength] where signal > 0 means buy
        """
        signals: Dict[str, float] = {}

        current_positions = sum(
            1 for data in self.datas if self.getposition(data).size > 0
        )

        for data in self.datas:
            ticker = data._name

            # Skip if already at max positions
            if current_positions >= self.p.max_positions:
                break

            # Skip if already in position
            if self.getposition(data).size > 0:
                continue

            # Check regime (ADX filter)
            if not self._is_ranging_regime(ticker):
                continue

            # Check for dip
            dip_return = self._calculate_dip(data)
            if dip_return is not None and dip_return <= self.p.dip_threshold:
                # Signal strength based on dip magnitude
                signal_strength = abs(dip_return) / abs(self.p.dip_threshold)
                signals[ticker] = min(1.0, signal_strength)
                current_positions += 1

                if self.p.verbose:
                    adx_val = self._get_adx_value(ticker)
                    print(
                        f"[{self.datas[0].datetime.date(0)}] {ticker}: "
                        f"ADX={adx_val:.1f}, Dip={dip_return * 100:.2f}% -> BUY SIGNAL"
                    )

        return signals

    def _is_ranging_regime(self, ticker: str) -> bool:
        """Check if ADX indicates ranging (non-trending) market."""
        adx_val = self._get_adx_value(ticker)

        if adx_val is None:
            return False

        return adx_val < self.p.adx_threshold

    def _get_adx_value(self, ticker: str) -> Optional[float]:
        """Get current ADX value for ticker."""
        adx = self._adx_indicators.get(ticker)

        if adx is None or len(adx) == 0:
            return None

        # Access the adx line from the indicator
        try:
            return adx.adx[0]
        except (AttributeError, IndexError):
            return None

    def _calculate_dip(self, data: bt.AbstractDataBase) -> Optional[float]:
        """Calculate short-term return to identify dips."""
        if len(data) < self.p.lookback_days + 1:
            return None

        current_close = data.close[0]
        past_close = data.close[-self.p.lookback_days]

        if past_close <= 0:
            return None

        return (current_close - past_close) / past_close

    def get_position_size(self, data: bt.AbstractDataBase, signal: float) -> int:
        """
        Calculate position size with volatility consideration.

        More conservative sizing for crypto due to higher volatility.
        """
        # Get available cash
        cash = self.broker.getcash()

        # Max allocation per position (e.g., 30% per crypto position)
        max_per_position = 0.30
        allocation = cash * max_per_position * signal

        # Calculate units
        current_price = data.close[0]
        if current_price <= 0:
            return 0

        # Round down for safety
        units = int(allocation / current_price)
        return max(0, units)
