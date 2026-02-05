"""
Performance reporting for Hybrid Trader.

Parses the trade audit trail (trades.csv) to calculate:
- Total PnL
- Win Rate
- Trade Count
- Per-symbol statistics
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Trade:
    """Represents a single trade from the audit trail."""

    timestamp: datetime
    symbol: str
    side: str  # BUY or SELL
    qty: Decimal
    price: Decimal
    order_id: str
    status: str


@dataclass
class PositionPnL:
    """Tracks PnL for a position (matched buy/sell pairs)."""

    symbol: str
    buy_qty: Decimal
    buy_price: Decimal
    sell_qty: Decimal
    sell_price: Decimal
    realized_pnl: Decimal
    is_closed: bool


@dataclass
class PerformanceReport:
    """Summary of trading performance."""

    total_trades: int
    buy_trades: int
    sell_trades: int
    total_volume: Decimal
    realized_pnl: Decimal
    winning_trades: int
    losing_trades: int
    win_rate: float
    symbols_traded: int
    per_symbol: Dict[str, dict]
    start_date: Optional[datetime]
    end_date: Optional[datetime]


def parse_trades(filepath: str = "trades.csv") -> List[Trade]:
    """
    Parse trades from CSV file.

    Args:
        filepath: Path to trades.csv

    Returns:
        List of Trade objects
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Trade log not found: {filepath}")

    trades: List[Trade] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                trade = Trade(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    symbol=row["symbol"],
                    side=row["side"],
                    qty=Decimal(row["qty"]),
                    price=Decimal(row["price"]) if row["price"] != "0" else Decimal("0"),
                    order_id=row["order_id"],
                    status=row["status"],
                )
                trades.append(trade)
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping malformed row: {row} ({e})")

    return trades


def calculate_pnl(trades: List[Trade]) -> Dict[str, PositionPnL]:
    """
    Calculate PnL by matching buy/sell trades per symbol.

    Uses FIFO (First In, First Out) matching.

    Args:
        trades: List of trades sorted by timestamp

    Returns:
        Dict mapping symbol -> PositionPnL
    """
    # Track open positions per symbol (FIFO queue of buys)
    open_positions: Dict[str, List[Trade]] = defaultdict(list)
    realized_pnl: Dict[str, Decimal] = defaultdict(Decimal)
    total_buy_qty: Dict[str, Decimal] = defaultdict(Decimal)
    total_sell_qty: Dict[str, Decimal] = defaultdict(Decimal)
    total_buy_value: Dict[str, Decimal] = defaultdict(Decimal)
    total_sell_value: Dict[str, Decimal] = defaultdict(Decimal)

    for trade in sorted(trades, key=lambda t: t.timestamp):
        symbol = trade.symbol

        if trade.side == "BUY":
            open_positions[symbol].append(trade)
            total_buy_qty[symbol] += trade.qty
            total_buy_value[symbol] += trade.qty * trade.price

        elif trade.side == "SELL":
            total_sell_qty[symbol] += trade.qty
            total_sell_value[symbol] += trade.qty * trade.price

            # Match against open positions (FIFO)
            remaining_qty = trade.qty
            while remaining_qty > 0 and open_positions[symbol]:
                open_trade = open_positions[symbol][0]

                if open_trade.qty <= remaining_qty:
                    # Close entire position
                    pnl = (trade.price - open_trade.price) * open_trade.qty
                    realized_pnl[symbol] += pnl
                    remaining_qty -= open_trade.qty
                    open_positions[symbol].pop(0)
                else:
                    # Partial close
                    pnl = (trade.price - open_trade.price) * remaining_qty
                    realized_pnl[symbol] += pnl
                    # Reduce the open position
                    open_positions[symbol][0] = Trade(
                        timestamp=open_trade.timestamp,
                        symbol=open_trade.symbol,
                        side=open_trade.side,
                        qty=open_trade.qty - remaining_qty,
                        price=open_trade.price,
                        order_id=open_trade.order_id,
                        status=open_trade.status,
                    )
                    remaining_qty = Decimal("0")

    # Build result
    result: Dict[str, PositionPnL] = {}
    all_symbols = set(total_buy_qty.keys()) | set(total_sell_qty.keys())

    for symbol in all_symbols:
        buy_qty = total_buy_qty[symbol]
        sell_qty = total_sell_qty[symbol]
        avg_buy = total_buy_value[symbol] / buy_qty if buy_qty > 0 else Decimal("0")
        avg_sell = total_sell_value[symbol] / sell_qty if sell_qty > 0 else Decimal("0")

        result[symbol] = PositionPnL(
            symbol=symbol,
            buy_qty=buy_qty,
            buy_price=avg_buy,
            sell_qty=sell_qty,
            sell_price=avg_sell,
            realized_pnl=realized_pnl[symbol],
            is_closed=len(open_positions.get(symbol, [])) == 0,
        )

    return result


def generate_report(trades: List[Trade]) -> PerformanceReport:
    """
    Generate a full performance report from trades.

    Args:
        trades: List of trades

    Returns:
        PerformanceReport with all statistics
    """
    if not trades:
        return PerformanceReport(
            total_trades=0,
            buy_trades=0,
            sell_trades=0,
            total_volume=Decimal("0"),
            realized_pnl=Decimal("0"),
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            symbols_traded=0,
            per_symbol={},
            start_date=None,
            end_date=None,
        )

    # Basic counts
    buy_trades = [t for t in trades if t.side == "BUY"]
    sell_trades = [t for t in trades if t.side == "SELL"]

    total_volume = sum((t.qty * t.price for t in trades), Decimal("0"))

    # PnL calculation
    pnl_by_symbol = calculate_pnl(trades)
    total_pnl = sum((p.realized_pnl for p in pnl_by_symbol.values()), Decimal("0"))

    # Win/loss counting (by symbol)
    winning = sum(1 for p in pnl_by_symbol.values() if p.realized_pnl > 0)
    losing = sum(1 for p in pnl_by_symbol.values() if p.realized_pnl < 0)

    total_closed = winning + losing
    win_rate = (winning / total_closed * 100) if total_closed > 0 else 0.0

    # Per-symbol stats
    per_symbol = {
        symbol: {
            "buy_qty": float(p.buy_qty),
            "sell_qty": float(p.sell_qty),
            "avg_buy_price": float(p.buy_price),
            "avg_sell_price": float(p.sell_price),
            "realized_pnl": float(p.realized_pnl),
            "is_closed": p.is_closed,
        }
        for symbol, p in pnl_by_symbol.items()
    }

    # Date range
    timestamps = [t.timestamp for t in trades]

    return PerformanceReport(
        total_trades=len(trades),
        buy_trades=len(buy_trades),
        sell_trades=len(sell_trades),
        total_volume=total_volume,
        realized_pnl=total_pnl,
        winning_trades=winning,
        losing_trades=losing,
        win_rate=win_rate,
        symbols_traded=len(pnl_by_symbol),
        per_symbol=per_symbol,
        start_date=min(timestamps),
        end_date=max(timestamps),
    )


def print_report(report: PerformanceReport) -> None:
    """
    Print a formatted performance report to console.

    Args:
        report: PerformanceReport to display
    """
    print("\n" + "=" * 60)
    print("           HYBRID TRADER PERFORMANCE REPORT")
    print("=" * 60)

    if report.total_trades == 0:
        print("\nNo trades found in the log.")
        return

    print(f"\nPeriod: {report.start_date} to {report.end_date}")

    print("\n--- SUMMARY ---")
    print(f"Total Trades:     {report.total_trades}")
    print(f"  Buy Orders:     {report.buy_trades}")
    print(f"  Sell Orders:    {report.sell_trades}")
    print(f"Total Volume:     ${report.total_volume:,.2f}")
    print(f"Symbols Traded:   {report.symbols_traded}")

    print("\n--- PERFORMANCE ---")
    pnl_color = "\033[92m" if report.realized_pnl >= 0 else "\033[91m"
    reset_color = "\033[0m"
    print(f"Realized PnL:     {pnl_color}${report.realized_pnl:,.2f}{reset_color}")
    print(f"Winning Symbols:  {report.winning_trades}")
    print(f"Losing Symbols:   {report.losing_trades}")
    print(f"Win Rate:         {report.win_rate:.1f}%")

    if report.per_symbol:
        print("\n--- PER SYMBOL ---")
        print(f"{'Symbol':<12} {'PnL':>12} {'Buy Qty':>10} {'Sell Qty':>10} {'Status':<8}")
        print("-" * 60)

        sorted_symbols = sorted(
            report.per_symbol.items(),
            key=lambda x: x[1]["realized_pnl"],
            reverse=True,
        )

        for symbol, stats in sorted_symbols:
            status = "Closed" if stats["is_closed"] else "Open"
            pnl = stats["realized_pnl"]
            pnl_str = f"${pnl:,.2f}"
            print(
                f"{symbol:<12} {pnl_str:>12} {stats['buy_qty']:>10.2f} "
                f"{stats['sell_qty']:>10.2f} {status:<8}"
            )

    print("\n" + "=" * 60)


def main() -> None:
    """Main entry point for reporting script."""
    parser = argparse.ArgumentParser(
        description="Generate performance report from trade log"
    )
    parser.add_argument(
        "--file",
        "-f",
        default="trades.csv",
        help="Path to trades.csv (default: trades.csv)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON instead of formatted text",
    )

    args = parser.parse_args()

    try:
        trades = parse_trades(args.file)
        report = generate_report(trades)

        if args.json:
            import json

            output = {
                "total_trades": report.total_trades,
                "buy_trades": report.buy_trades,
                "sell_trades": report.sell_trades,
                "total_volume": float(report.total_volume),
                "realized_pnl": float(report.realized_pnl),
                "winning_trades": report.winning_trades,
                "losing_trades": report.losing_trades,
                "win_rate": report.win_rate,
                "symbols_traded": report.symbols_traded,
                "per_symbol": report.per_symbol,
                "start_date": report.start_date.isoformat() if report.start_date else None,
                "end_date": report.end_date.isoformat() if report.end_date else None,
            }
            print(json.dumps(output, indent=2))
        else:
            print_report(report)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
