"""
Tests for Phase 3 Execution Bridge.

All tests use mocked Alpaca API - NO real orders are ever placed.
"""
from __future__ import annotations

import logging
import pytest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone

import sys

sys.path.insert(0, str(__file__).replace("/tests/test_execution.py", "/src"))

from execution.base import ExecutorBase, ExecutorError, SafetyError
from execution.models import (
    AccountInfo,
    PositionInfo,
    OrderRequest,
    OrderResult,
    OrderSideEnum,
    OrderTypeEnum,
    OrderStatusEnum,
    AssetClassEnum,
)
from execution.router import OrderRouter, SignalConfig, PendingOrder


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_account_info() -> AccountInfo:
    """Create mock account info."""
    return AccountInfo(
        cash=Decimal("100000"),
        buying_power=Decimal("200000"),
        portfolio_value=Decimal("150000"),
        equity=Decimal("150000"),
        is_trading_blocked=False,
        is_pattern_day_trader=False,
    )


@pytest.fixture
def mock_position_info() -> PositionInfo:
    """Create mock position info."""
    return PositionInfo(
        symbol="AAPL",
        qty=Decimal("100"),
        avg_entry_price=Decimal("150.00"),
        market_value=Decimal("15500"),
        unrealized_pnl=Decimal("500"),
        unrealized_pnl_pct=Decimal("0.0333"),
        asset_class=AssetClassEnum.EQUITY,
        qty_available=Decimal("100"),
    )


@pytest.fixture
def mock_order_result() -> OrderResult:
    """Create mock order result."""
    return OrderResult(
        order_id="test-order-123",
        client_order_id="client-123",
        symbol="AAPL",
        qty=Decimal("10"),
        filled_qty=Decimal("10"),
        filled_avg_price=Decimal("155.00"),
        status=OrderStatusEnum.FILLED,
        created_at=datetime.now(timezone.utc),
        filled_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_executor(
    mock_account_info: AccountInfo,
    mock_position_info: PositionInfo,
    mock_order_result: OrderResult,
) -> Mock:
    """Create mock executor with standard responses."""
    executor = Mock(spec=ExecutorBase)
    executor.get_account_info.return_value = mock_account_info
    executor.get_positions.return_value = {}
    executor.get_position.return_value = None
    executor.submit_order.return_value = mock_order_result
    executor.get_order.return_value = mock_order_result
    executor.get_open_orders.return_value = []
    executor.cancel_order.return_value = True
    executor.close_position.return_value = mock_order_result
    executor.get_asset_class.side_effect = lambda s: (
        AssetClassEnum.CRYPTO if "-USD" in s else AssetClassEnum.EQUITY
    )
    return executor


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test data transfer objects."""

    def test_account_info_immutable(self, mock_account_info: AccountInfo) -> None:
        """AccountInfo should be frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            mock_account_info.cash = Decimal("50000")  # type: ignore

    def test_position_info_creation(self) -> None:
        """Test PositionInfo creation with all fields."""
        pos = PositionInfo(
            symbol="BTC-USD",
            qty=Decimal("0.5"),
            avg_entry_price=Decimal("45000"),
            market_value=Decimal("23000"),
            unrealized_pnl=Decimal("500"),
            unrealized_pnl_pct=Decimal("0.0222"),
            asset_class=AssetClassEnum.CRYPTO,
            qty_available=Decimal("0.5"),
        )
        assert pos.symbol == "BTC-USD"
        assert pos.asset_class == AssetClassEnum.CRYPTO

    def test_order_result_is_terminal(self, mock_order_result: OrderResult) -> None:
        """Test terminal state detection."""
        assert mock_order_result.is_terminal  # FILLED is terminal

        # Pending is not terminal
        pending = OrderResult(
            order_id="pending-123",
            client_order_id=None,
            symbol="AAPL",
            qty=Decimal("10"),
            filled_qty=Decimal("0"),
            filled_avg_price=None,
            status=OrderStatusEnum.SUBMITTED,
            created_at=datetime.now(timezone.utc),
        )
        assert not pending.is_terminal

    def test_order_result_is_partial(self) -> None:
        """Test partial fill detection."""
        partial = OrderResult(
            order_id="partial-123",
            client_order_id=None,
            symbol="AAPL",
            qty=Decimal("100"),
            filled_qty=Decimal("50"),
            filled_avg_price=Decimal("150"),
            status=OrderStatusEnum.PARTIALLY_FILLED,
            created_at=datetime.now(timezone.utc),
        )
        assert partial.is_partial

    def test_order_request_defaults(self) -> None:
        """Test OrderRequest default values."""
        request = OrderRequest(
            symbol="AAPL",
            qty=Decimal("10"),
            side=OrderSideEnum.BUY,
        )
        assert request.order_type == OrderTypeEnum.MARKET
        assert request.limit_price is None


# =============================================================================
# Executor Base Tests
# =============================================================================


class TestExecutorBase:
    """Test abstract executor interface."""

    def test_get_asset_class_equity(self) -> None:
        """Test equity detection."""

        # Create concrete implementation for testing
        class TestExecutor(ExecutorBase):
            def get_account_info(self):
                pass

            def get_positions(self):
                pass

            def get_position(self, symbol):
                pass

            def submit_order(self, request):
                pass

            def get_order(self, order_id):
                pass

            def get_open_orders(self):
                pass

            def cancel_order(self, order_id):
                pass

            def close_position(self, symbol):
                pass

            def close_all_positions(self):
                pass

        executor = TestExecutor()
        assert executor.get_asset_class("AAPL") == AssetClassEnum.EQUITY
        assert executor.get_asset_class("MSFT") == AssetClassEnum.EQUITY
        assert executor.get_asset_class("SPY") == AssetClassEnum.EQUITY

    def test_get_asset_class_crypto(self) -> None:
        """Test crypto detection."""

        class TestExecutor(ExecutorBase):
            def get_account_info(self):
                pass

            def get_positions(self):
                pass

            def get_position(self, symbol):
                pass

            def submit_order(self, request):
                pass

            def get_order(self, order_id):
                pass

            def get_open_orders(self):
                pass

            def cancel_order(self, order_id):
                pass

            def close_position(self, symbol):
                pass

            def close_all_positions(self):
                pass

        executor = TestExecutor()
        assert executor.get_asset_class("BTC-USD") == AssetClassEnum.CRYPTO
        assert executor.get_asset_class("ETH-USD") == AssetClassEnum.CRYPTO
        assert executor.get_asset_class("BTC/USD") == AssetClassEnum.CRYPTO
        assert executor.get_asset_class("BTCUSDT") == AssetClassEnum.CRYPTO


# =============================================================================
# AlpacaExecutor Tests (Mocked)
# =============================================================================


class TestAlpacaExecutorSafety:
    """Test AlpacaExecutor safety mechanisms."""

    def test_safety_error_live_without_flag(self) -> None:
        """Live trading without explicit flag should raise SafetyError."""
        with patch("execution.alpaca.TradingClient"):
            from execution.alpaca import AlpacaExecutor

            with pytest.raises(SafetyError) as exc_info:
                AlpacaExecutor(
                    api_key="test",
                    secret_key="test",
                    paper=False,
                    live_trading_enabled=False,
                )

            assert "live_trading_enabled=True" in str(exc_info.value)

    def test_paper_mode_allowed_without_flag(self) -> None:
        """Paper trading should work without explicit flag."""
        with patch("execution.alpaca.TradingClient"):
            from execution.alpaca import AlpacaExecutor

            executor = AlpacaExecutor(
                api_key="test",
                secret_key="test",
                paper=True,
            )
            assert executor.is_paper

    def test_live_mode_with_explicit_flag(self) -> None:
        """Live trading should work with explicit flag."""
        with patch("execution.alpaca.TradingClient"):
            from execution.alpaca import AlpacaExecutor

            executor = AlpacaExecutor(
                api_key="test",
                secret_key="test",
                paper=False,
                live_trading_enabled=True,
            )
            assert not executor.is_paper


class TestAlpacaExecutorOperations:
    """Test AlpacaExecutor operations with mocked API."""

    @pytest.fixture
    def executor_with_mock_client(self):
        """Create executor with fully mocked client."""
        with patch("execution.alpaca.TradingClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            from execution.alpaca import AlpacaExecutor

            executor = AlpacaExecutor(
                api_key="test",
                secret_key="test",
                paper=True,
            )

            yield executor, mock_client

    def test_get_account_info(self, executor_with_mock_client) -> None:
        """Test account info retrieval."""
        executor, mock_client = executor_with_mock_client

        # Setup mock response
        mock_account = Mock()
        mock_account.cash = "100000"
        mock_account.buying_power = "200000"
        mock_account.portfolio_value = "150000"
        mock_account.equity = "150000"
        mock_account.trading_blocked = False
        mock_account.pattern_day_trader = False
        mock_client.get_account.return_value = mock_account

        account = executor.get_account_info()

        assert account.cash == Decimal("100000")
        assert account.buying_power == Decimal("200000")

    def test_get_positions(self, executor_with_mock_client) -> None:
        """Test positions retrieval."""
        executor, mock_client = executor_with_mock_client

        # Setup mock response
        mock_pos = Mock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "100"
        mock_pos.avg_entry_price = "150.00"
        mock_pos.market_value = "15500"
        mock_pos.unrealized_pl = "500"
        mock_pos.unrealized_plpc = "0.0333"
        mock_pos.asset_class = None
        mock_pos.qty_available = "100"
        mock_client.get_all_positions.return_value = [mock_pos]

        positions = executor.get_positions()

        assert "AAPL" in positions
        assert positions["AAPL"].qty == Decimal("100")

    def test_submit_order(self, executor_with_mock_client) -> None:
        """Test order submission."""
        executor, mock_client = executor_with_mock_client

        # Setup mock response
        mock_order = Mock()
        mock_order.id = "order-123"
        mock_order.client_order_id = "client-123"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_order.status = Mock()
        mock_order.created_at = datetime.now(timezone.utc)
        mock_order.filled_at = None
        mock_client.submit_order.return_value = mock_order

        request = OrderRequest(
            symbol="AAPL",
            qty=Decimal("10"),
            side=OrderSideEnum.BUY,
        )

        result = executor.submit_order(request)

        assert result.symbol == "AAPL"
        mock_client.submit_order.assert_called_once()

    def test_submit_order_crypto_uses_gtc(self, executor_with_mock_client) -> None:
        """Crypto orders must use GTC time-in-force (not DAY)."""
        from alpaca.trading.enums import TimeInForce
        executor, mock_client = executor_with_mock_client

        # Setup mock response
        mock_order = Mock()
        mock_order.id = "order-456"
        mock_order.client_order_id = "client-456"
        mock_order.symbol = "ETH/USD"
        mock_order.qty = "1.5"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_order.status = Mock()
        mock_order.created_at = datetime.now(timezone.utc)
        mock_order.filled_at = None
        mock_client.submit_order.return_value = mock_order

        # Submit crypto SELL order (the failing case in paper trading)
        request = OrderRequest(
            symbol="ETH-USD",  # Internal format
            qty=Decimal("1.5"),
            side=OrderSideEnum.SELL,
        )

        executor.submit_order(request)

        # Verify the order was submitted with GTC (not DAY)
        call_args = mock_client.submit_order.call_args
        submitted_request = call_args[0][0]  # First positional arg
        assert submitted_request.time_in_force == TimeInForce.GTC
        assert submitted_request.symbol == "ETH/USD"  # Translated to Alpaca format

    def test_submit_order_equity_uses_day(self, executor_with_mock_client) -> None:
        """Equity orders should use DAY time-in-force."""
        from alpaca.trading.enums import TimeInForce
        executor, mock_client = executor_with_mock_client

        mock_order = Mock()
        mock_order.id = "order-789"
        mock_order.client_order_id = "client-789"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_order.status = Mock()
        mock_order.created_at = datetime.now(timezone.utc)
        mock_order.filled_at = None
        mock_client.submit_order.return_value = mock_order

        request = OrderRequest(
            symbol="AAPL",
            qty=Decimal("10"),
            side=OrderSideEnum.BUY,
        )

        executor.submit_order(request)

        call_args = mock_client.submit_order.call_args
        submitted_request = call_args[0][0]
        assert submitted_request.time_in_force == TimeInForce.DAY


# =============================================================================
# OrderRouter Tests
# =============================================================================


class TestOrderRouter:
    """Test order routing logic."""

    def test_execute_empty_signals(self, mock_executor: Mock) -> None:
        """Empty signals should produce no orders."""
        router = OrderRouter(mock_executor)
        results = router.execute_signals({})

        assert len(results) == 0
        mock_executor.submit_order.assert_not_called()

    def test_execute_signals_skips_existing_positions(
        self, mock_executor: Mock, mock_position_info: PositionInfo
    ) -> None:
        """Signals for existing positions should be skipped."""
        mock_executor.get_positions.return_value = {"AAPL": mock_position_info}

        router = OrderRouter(mock_executor)

        # Signal for AAPL (which we already have) should be skipped
        results = router.execute_signals({"AAPL": 0.8})

        assert len(results) == 0

    def test_execute_signals_creates_orders(self, mock_executor: Mock) -> None:
        """Valid signals should create orders."""

        def price_fetcher(symbol: str) -> Decimal:
            return Decimal("150.00")

        router = OrderRouter(mock_executor, price_fetcher=price_fetcher)

        results = router.execute_signals({"AAPL": 0.8})

        assert len(results) == 1
        mock_executor.submit_order.assert_called_once()

    def test_signal_config_max_position(self, mock_executor: Mock) -> None:
        """Max position percentage should be respected."""
        config = SignalConfig(
            max_position_pct=Decimal("0.05"),
            min_order_value=Decimal("100"),
        )

        def price_fetcher(symbol: str) -> Decimal:
            return Decimal("150.00")

        router = OrderRouter(mock_executor, config=config, price_fetcher=price_fetcher)

        results = router.execute_signals({"AAPL": 1.0})

        # Check the order was sized correctly
        call_args = mock_executor.submit_order.call_args
        request = call_args[0][0]

        # Max 5% of $150,000 portfolio = $7,500
        # At $150/share = 50 shares max
        assert request.qty <= Decimal("50")

    def test_dry_run_no_orders(self, mock_executor: Mock) -> None:
        """Dry run should not submit orders."""

        def price_fetcher(symbol: str) -> Decimal:
            return Decimal("150.00")

        router = OrderRouter(mock_executor, price_fetcher=price_fetcher)

        results = router.execute_signals({"AAPL": 0.8}, dry_run=True)

        assert len(results) == 0
        mock_executor.submit_order.assert_not_called()

    def test_asset_class_separation(self, mock_executor: Mock) -> None:
        """Crypto and equity signals should be processed separately."""

        def price_fetcher(symbol: str) -> Decimal:
            if "BTC" in symbol:
                return Decimal("45000")
            return Decimal("150")

        router = OrderRouter(mock_executor, price_fetcher=price_fetcher)

        signals = {
            "AAPL": 0.5,
            "BTC-USD": 0.5,
        }

        results = router.execute_signals(signals)

        # Both orders should be submitted
        assert mock_executor.submit_order.call_count == 2


class TestOrderRouterPendingOrders:
    """Test pending order tracking."""

    def test_pending_order_tracking(self, mock_executor: Mock) -> None:
        """Submitted orders should be tracked as pending."""
        # Make get_order return a non-terminal (SUBMITTED) order
        submitted_order = OrderResult(
            order_id="test-order-123",
            client_order_id="client-123",
            symbol="AAPL",
            qty=Decimal("10"),
            filled_qty=Decimal("0"),
            filled_avg_price=None,
            status=OrderStatusEnum.SUBMITTED,  # Non-terminal
            created_at=datetime.now(timezone.utc),
        )
        mock_executor.get_order.return_value = submitted_order

        def price_fetcher(symbol: str) -> Decimal:
            return Decimal("150")

        router = OrderRouter(mock_executor, price_fetcher=price_fetcher)
        router.execute_signals({"AAPL": 0.8})

        pending = router.get_pending_orders()
        assert len(pending) == 1
        assert pending[0].symbol == "AAPL"

    def test_cancel_pending_orders(self, mock_executor: Mock) -> None:
        """Cancellation should remove from pending."""

        def price_fetcher(symbol: str) -> Decimal:
            return Decimal("150")

        router = OrderRouter(mock_executor, price_fetcher=price_fetcher)
        router.execute_signals({"AAPL": 0.8})

        cancelled = router.cancel_pending_orders()

        assert cancelled == 1
        assert len(router.get_pending_orders()) == 0


class TestPartialFills:
    """Test partial fill handling."""

    def test_handle_partial_fill_detection(self, mock_executor: Mock) -> None:
        """Partial fills should be detected correctly."""
        partial_result = OrderResult(
            order_id="partial-123",
            client_order_id=None,
            symbol="AAPL",
            qty=Decimal("100"),
            filled_qty=Decimal("50"),
            filled_avg_price=Decimal("150"),
            status=OrderStatusEnum.PARTIALLY_FILLED,
            created_at=datetime.now(timezone.utc),
        )
        mock_executor.get_order.return_value = partial_result

        router = OrderRouter(mock_executor)

        result = router.handle_partial_fill("partial-123")

        assert result is not None
        assert result.is_partial
        assert result.filled_qty == Decimal("50")


# =============================================================================
# Integration Tests (All Mocked)
# =============================================================================


class TestIntegration:
    """Integration tests with full pipeline."""

    def test_full_signal_to_order_pipeline(self, mock_executor: Mock) -> None:
        """Test complete flow from signal to order."""

        def price_fetcher(symbol: str) -> Decimal:
            prices = {"AAPL": Decimal("150"), "MSFT": Decimal("300")}
            return prices.get(symbol, Decimal("100"))

        router = OrderRouter(mock_executor, price_fetcher=price_fetcher)

        # Simulate strategy signals
        signals = {
            "AAPL": 0.8,
            "MSFT": 0.5,
        }

        results = router.execute_signals(signals)

        # Should have 2 orders
        assert mock_executor.submit_order.call_count == 2

    def test_portfolio_sync(
        self, mock_executor: Mock, mock_position_info: PositionInfo
    ) -> None:
        """Test portfolio synchronization."""
        mock_executor.get_positions.return_value = {"AAPL": mock_position_info}

        router = OrderRouter(mock_executor)
        positions = router.sync_portfolio()

        assert "AAPL" in positions
        assert positions["AAPL"].qty == Decimal("100")


# =============================================================================
# Portfolio Reconciliation Tests (Phase 6.1)
# =============================================================================


class TestPortfolioReconciliation:
    """Test the portfolio reconciliation (Target vs Actual) logic."""

    @pytest.fixture
    def aapl_position(self) -> PositionInfo:
        """Create AAPL position fixture."""
        return PositionInfo(
            symbol="AAPL",
            qty=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            market_value=Decimal("15500"),
            unrealized_pnl=Decimal("500"),
            unrealized_pnl_pct=Decimal("0.0333"),
            asset_class=AssetClassEnum.EQUITY,
            qty_available=Decimal("100"),
        )

    @pytest.fixture
    def msft_position(self) -> PositionInfo:
        """Create MSFT position fixture."""
        return PositionInfo(
            symbol="MSFT",
            qty=Decimal("50"),
            avg_entry_price=Decimal("300.00"),
            market_value=Decimal("16000"),
            unrealized_pnl=Decimal("1000"),
            unrealized_pnl_pct=Decimal("0.0667"),
            asset_class=AssetClassEnum.EQUITY,
            qty_available=Decimal("50"),
        )

    @pytest.fixture
    def btc_position(self) -> PositionInfo:
        """Create BTC-USD position fixture."""
        return PositionInfo(
            symbol="BTC-USD",
            qty=Decimal("0.5"),
            avg_entry_price=Decimal("45000.00"),
            market_value=Decimal("23000"),
            unrealized_pnl=Decimal("500"),
            unrealized_pnl_pct=Decimal("0.0222"),
            asset_class=AssetClassEnum.CRYPTO,
            qty_available=Decimal("0.5"),
        )

    def test_sell_when_signal_disappears(
        self, mock_executor: Mock, aapl_position: PositionInfo
    ) -> None:
        """Position should be sold when symbol disappears from signals."""
        # Setup: We hold AAPL
        mock_executor.get_positions.return_value = {"AAPL": aapl_position}

        router = OrderRouter(mock_executor)

        # Signal doesn't include AAPL at all
        signals: Dict[str, float] = {}

        results = router.execute_signals(signals)

        # Should submit SELL order for AAPL
        mock_executor.submit_order.assert_called_once()
        call_args = mock_executor.submit_order.call_args[0][0]
        assert call_args.symbol == "AAPL"
        assert call_args.side == OrderSideEnum.SELL
        assert call_args.qty == Decimal("100")

    def test_sell_when_signal_drops_to_zero(
        self, mock_executor: Mock, aapl_position: PositionInfo
    ) -> None:
        """Position should be sold when signal strength drops to 0."""
        mock_executor.get_positions.return_value = {"AAPL": aapl_position}

        router = OrderRouter(mock_executor)

        # Signal is explicitly 0 (exit signal)
        signals = {"AAPL": 0.0}

        results = router.execute_signals(signals)

        # Should submit SELL order
        mock_executor.submit_order.assert_called_once()
        call_args = mock_executor.submit_order.call_args[0][0]
        assert call_args.symbol == "AAPL"
        assert call_args.side == OrderSideEnum.SELL
        assert call_args.qty == Decimal("100")

    def test_sell_when_signal_negative(
        self, mock_executor: Mock, aapl_position: PositionInfo
    ) -> None:
        """Position should be sold when signal is negative."""
        mock_executor.get_positions.return_value = {"AAPL": aapl_position}

        router = OrderRouter(mock_executor)

        # Negative signal (strong exit)
        signals = {"AAPL": -0.5}

        results = router.execute_signals(signals)

        # Should submit SELL order
        mock_executor.submit_order.assert_called_once()
        call_args = mock_executor.submit_order.call_args[0][0]
        assert call_args.side == OrderSideEnum.SELL

    def test_no_action_when_signal_positive_and_held(
        self, mock_executor: Mock, aapl_position: PositionInfo
    ) -> None:
        """No orders when position is held and signal remains positive."""
        mock_executor.get_positions.return_value = {"AAPL": aapl_position}

        router = OrderRouter(mock_executor)

        # Signal still positive for held position
        signals = {"AAPL": 0.8}

        results = router.execute_signals(signals)

        # No orders should be submitted (position maintained)
        mock_executor.submit_order.assert_not_called()

    def test_reconciliation_multiple_symbols(
        self,
        mock_executor: Mock,
        aapl_position: PositionInfo,
        msft_position: PositionInfo,
    ) -> None:
        """Test reconciliation with multiple symbols: some close, some open, some hold."""
        # We hold AAPL and MSFT
        mock_executor.get_positions.return_value = {
            "AAPL": aapl_position,
            "MSFT": msft_position,
        }

        def price_fetcher(symbol: str) -> Decimal:
            return Decimal("200")

        router = OrderRouter(mock_executor, price_fetcher=price_fetcher)

        # Signals: Keep AAPL, exit MSFT, enter GOOGL
        signals = {
            "AAPL": 0.8,  # Keep
            "MSFT": 0.0,  # Exit
            "GOOGL": 0.6,  # New entry
        }

        results = router.execute_signals(signals)

        # Should have 2 orders: SELL MSFT, BUY GOOGL
        assert mock_executor.submit_order.call_count == 2

        # Verify order types
        calls = mock_executor.submit_order.call_args_list
        symbols_and_sides = [(c[0][0].symbol, c[0][0].side) for c in calls]

        assert ("MSFT", OrderSideEnum.SELL) in symbols_and_sides
        assert ("GOOGL", OrderSideEnum.BUY) in symbols_and_sides
        # AAPL should NOT be in any order (held with positive signal)
        assert not any(s == "AAPL" for s, _ in symbols_and_sides)

    def test_reconciliation_close_all_exit_signals(
        self,
        mock_executor: Mock,
        aapl_position: PositionInfo,
        msft_position: PositionInfo,
    ) -> None:
        """All positions closed when all signals are zero."""
        mock_executor.get_positions.return_value = {
            "AAPL": aapl_position,
            "MSFT": msft_position,
        }

        router = OrderRouter(mock_executor)

        # All signals zero - full exit
        signals = {"AAPL": 0.0, "MSFT": 0.0}

        results = router.execute_signals(signals)

        # Should have 2 SELL orders
        assert mock_executor.submit_order.call_count == 2
        for call in mock_executor.submit_order.call_args_list:
            assert call[0][0].side == OrderSideEnum.SELL

    def test_closes_before_opens(
        self,
        mock_executor: Mock,
        aapl_position: PositionInfo,
    ) -> None:
        """Close orders should be submitted before open orders."""
        mock_executor.get_positions.return_value = {"AAPL": aapl_position}

        def price_fetcher(symbol: str) -> Decimal:
            return Decimal("200")

        router = OrderRouter(mock_executor, price_fetcher=price_fetcher)

        # Exit AAPL, enter MSFT
        signals = {"AAPL": 0.0, "MSFT": 0.8}

        results = router.execute_signals(signals)

        # Should have 2 orders
        assert mock_executor.submit_order.call_count == 2

        # First call should be SELL (close), second should be BUY (open)
        first_call = mock_executor.submit_order.call_args_list[0][0][0]
        second_call = mock_executor.submit_order.call_args_list[1][0][0]

        assert first_call.side == OrderSideEnum.SELL
        assert first_call.symbol == "AAPL"
        assert second_call.side == OrderSideEnum.BUY
        assert second_call.symbol == "MSFT"

    def test_crypto_position_close(
        self, mock_executor: Mock, btc_position: PositionInfo
    ) -> None:
        """Crypto positions should close correctly with fractional quantities."""
        mock_executor.get_positions.return_value = {"BTC-USD": btc_position}

        router = OrderRouter(mock_executor)

        # Exit crypto position
        signals = {"BTC-USD": 0.0}

        results = router.execute_signals(signals)

        # Should sell exact fractional quantity
        mock_executor.submit_order.assert_called_once()
        call_args = mock_executor.submit_order.call_args[0][0]
        assert call_args.symbol == "BTC-USD"
        assert call_args.side == OrderSideEnum.SELL
        assert call_args.qty == Decimal("0.5")

    def test_dry_run_logs_closes_and_opens(
        self,
        mock_executor: Mock,
        aapl_position: PositionInfo,
        caplog,
    ) -> None:
        """Dry run should log both close and open orders without executing."""
        mock_executor.get_positions.return_value = {"AAPL": aapl_position}

        def price_fetcher(symbol: str) -> Decimal:
            return Decimal("200")

        router = OrderRouter(mock_executor, price_fetcher=price_fetcher)

        signals = {"AAPL": 0.0, "MSFT": 0.8}

        with caplog.at_level(logging.INFO):
            results = router.execute_signals(signals, dry_run=True)

        # No orders submitted in dry run
        mock_executor.submit_order.assert_not_called()
        assert len(results) == 0

        # Logs should mention both close and open
        log_text = caplog.text
        assert "CLOSE" in log_text or "SELL" in log_text
        assert "OPEN" in log_text or "BUY" in log_text

    def test_zero_quantity_position_not_closed(
        self, mock_executor: Mock
    ) -> None:
        """Position with zero quantity should not generate close order."""
        # Edge case: position exists but qty is 0
        zero_position = PositionInfo(
            symbol="AAPL",
            qty=Decimal("0"),
            avg_entry_price=Decimal("150.00"),
            market_value=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            unrealized_pnl_pct=Decimal("0"),
            asset_class=AssetClassEnum.EQUITY,
            qty_available=Decimal("0"),
        )
        mock_executor.get_positions.return_value = {"AAPL": zero_position}

        router = OrderRouter(mock_executor)

        signals = {}  # No signals

        results = router.execute_signals(signals)

        # No orders for zero quantity
        mock_executor.submit_order.assert_not_called()

    def test_close_all_positions_method(
        self,
        mock_executor: Mock,
        aapl_position: PositionInfo,
        msft_position: PositionInfo,
    ) -> None:
        """Emergency close_all_positions should close everything."""
        mock_executor.get_positions.return_value = {
            "AAPL": aapl_position,
            "MSFT": msft_position,
        }

        router = OrderRouter(mock_executor)

        results = router.close_all_positions()

        # Should have 2 SELL orders
        assert mock_executor.submit_order.call_count == 2
        for call in mock_executor.submit_order.call_args_list:
            assert call[0][0].side == OrderSideEnum.SELL

    def test_close_all_positions_dry_run(
        self,
        mock_executor: Mock,
        aapl_position: PositionInfo,
    ) -> None:
        """close_all_positions with dry_run should not execute."""
        mock_executor.get_positions.return_value = {"AAPL": aapl_position}

        router = OrderRouter(mock_executor)

        results = router.close_all_positions(dry_run=True)

        mock_executor.submit_order.assert_not_called()
        assert len(results) == 0


class TestReconciliationEdgeCases:
    """Edge cases for portfolio reconciliation."""

    def test_empty_portfolio_positive_signals(self, mock_executor: Mock) -> None:
        """Starting from empty portfolio with positive signals = all buys."""
        mock_executor.get_positions.return_value = {}

        def price_fetcher(symbol: str) -> Decimal:
            return Decimal("100")

        router = OrderRouter(mock_executor, price_fetcher=price_fetcher)

        signals = {"AAPL": 0.5, "MSFT": 0.5}

        results = router.execute_signals(signals)

        # Should have 2 BUY orders
        assert mock_executor.submit_order.call_count == 2
        for call in mock_executor.submit_order.call_args_list:
            assert call[0][0].side == OrderSideEnum.BUY

    def test_full_portfolio_no_signals(
        self, mock_executor: Mock
    ) -> None:
        """Full portfolio with no signals = all sells."""
        pos = PositionInfo(
            symbol="AAPL",
            qty=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            unrealized_pnl_pct=Decimal("0"),
            asset_class=AssetClassEnum.EQUITY,
            qty_available=Decimal("100"),
        )
        mock_executor.get_positions.return_value = {"AAPL": pos}

        router = OrderRouter(mock_executor)

        # No signals = exit everything
        results = router.execute_signals({})

        assert mock_executor.submit_order.call_count == 1
        call_args = mock_executor.submit_order.call_args[0][0]
        assert call_args.side == OrderSideEnum.SELL

    def test_price_fetch_failure_skips_buy(self, mock_executor: Mock) -> None:
        """Failed price fetch should skip buy order but not crash."""
        mock_executor.get_positions.return_value = {}

        def failing_price_fetcher(symbol: str) -> Decimal:
            return Decimal("0")  # Invalid price

        router = OrderRouter(mock_executor, price_fetcher=failing_price_fetcher)

        signals = {"AAPL": 0.8}

        results = router.execute_signals(signals)

        # No orders due to price failure
        mock_executor.submit_order.assert_not_called()

    def test_executor_error_continues_other_orders(
        self, mock_executor: Mock
    ) -> None:
        """Executor error on one order should not prevent others."""
        pos1 = PositionInfo(
            symbol="AAPL",
            qty=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            unrealized_pnl_pct=Decimal("0"),
            asset_class=AssetClassEnum.EQUITY,
            qty_available=Decimal("100"),
        )
        pos2 = PositionInfo(
            symbol="MSFT",
            qty=Decimal("50"),
            avg_entry_price=Decimal("300.00"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            unrealized_pnl_pct=Decimal("0"),
            asset_class=AssetClassEnum.EQUITY,
            qty_available=Decimal("50"),
        )
        mock_executor.get_positions.return_value = {"AAPL": pos1, "MSFT": pos2}

        # First call fails, second succeeds
        mock_executor.submit_order.side_effect = [
            ExecutorError("API Error"),
            OrderResult(
                order_id="msft-123",
                client_order_id=None,
                symbol="MSFT",
                qty=Decimal("50"),
                filled_qty=Decimal("50"),
                filled_avg_price=Decimal("300"),
                status=OrderStatusEnum.FILLED,
                created_at=datetime.now(timezone.utc),
            ),
        ]

        router = OrderRouter(mock_executor)

        results = router.execute_signals({})  # Close all

        # Should have attempted both orders
        assert mock_executor.submit_order.call_count == 2
        # Only one succeeded
        assert len(results) == 1
        assert results[0].symbol == "MSFT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
