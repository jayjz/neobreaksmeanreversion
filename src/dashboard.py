"""
Hybrid Trader Dashboard - Real-time Performance Monitoring

A professional quant terminal built with Streamlit and Plotly.

Usage:
    streamlit run src/dashboard.py
    streamlit run src/dashboard.py -- --trades trades.csv --health /tmp/hybrid_trader.health
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting import parse_trades, generate_report, PerformanceReport, Trade


# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Hybrid Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional quant terminal look
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
    .stMetric label {
        color: #888 !important;
        font-size: 0.9rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    /* Positive/negative colors */
    .metric-positive { color: #00d26a !important; }
    .metric-negative { color: #ff4757 !important; }
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-top: 0;
    }
    /* Status indicator */
    .status-online {
        color: #00d26a;
        font-weight: 600;
    }
    .status-offline {
        color: #ff4757;
        font-weight: 600;
    }
    .status-stale {
        color: #ffa502;
        font-weight: 600;
    }
    /* Table styling */
    .dataframe {
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data(ttl=30)  # Refresh every 30 seconds
def load_trades(filepath: str) -> Optional[pd.DataFrame]:
    """Load trades from CSV into a DataFrame."""
    try:
        path = Path(filepath)
        if not path.exists():
            return None

        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        if df.empty:
            return None

        # Ensure numeric columns
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["value"] = df["qty"] * df["price"]

        return df
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        return None


@st.cache_data(ttl=30)
def load_report(filepath: str) -> Optional[PerformanceReport]:
    """Load and generate performance report."""
    try:
        trades = parse_trades(filepath)
        if not trades:
            return None
        return generate_report(trades)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return None


def check_system_health(health_file: str, stale_threshold_minutes: int = 10) -> dict:
    """Check system health based on heartbeat file."""
    path = Path(health_file)

    if not path.exists():
        return {
            "status": "offline",
            "message": "Heartbeat file not found",
            "last_update": None,
        }

    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(timezone.utc) - mtime

        if age > timedelta(minutes=stale_threshold_minutes):
            return {
                "status": "stale",
                "message": f"Last heartbeat {age.total_seconds() / 60:.1f} min ago",
                "last_update": mtime,
            }

        return {
            "status": "online",
            "message": f"Last heartbeat {age.total_seconds():.0f}s ago",
            "last_update": mtime,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "last_update": None,
        }


def calculate_cumulative_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative PnL from trades."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Sort by timestamp
    df = df.sort_values("timestamp").copy()

    # Calculate PnL per trade (simplified: assumes matched pairs)
    # For accurate PnL, we'd need FIFO matching, but this gives a reasonable approximation
    df["pnl"] = 0.0

    # Track positions per symbol
    positions: dict[str, dict[str, float]] = {}
    pnl_list: list[float] = []

    for _, row in df.iterrows():
        symbol = row["symbol"]
        side = row["side"]
        qty = float(row["qty"])
        price = float(row["price"])

        if symbol not in positions:
            positions[symbol] = {"qty": 0.0, "avg_price": 0.0}

        pos = positions[symbol]

        if side == "BUY":
            # Add to position
            total_qty = pos["qty"] + qty
            if total_qty > 0:
                pos["avg_price"] = (pos["qty"] * pos["avg_price"] + qty * price) / total_qty
            pos["qty"] = total_qty
            pnl_list.append(0.0)  # No realized PnL on buy
        else:  # SELL
            # Realize PnL
            if pos["qty"] > 0:
                realized = (price - pos["avg_price"]) * min(qty, pos["qty"])
                pos["qty"] -= qty
                pnl_list.append(realized)
            else:
                pnl_list.append(0.0)

    df["pnl"] = pnl_list
    df["cumulative_pnl"] = df["pnl"].cumsum()

    return df


# =============================================================================
# Visualization Components
# =============================================================================

def render_header(health_status: dict):
    """Render the main header with status indicator."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<p class="main-header">HYBRID TRADER</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Real-time Performance Dashboard</p>', unsafe_allow_html=True)

    with col2:
        status = health_status["status"]
        if status == "online":
            st.markdown(f'<p class="status-online">● SYSTEM ONLINE</p>', unsafe_allow_html=True)
        elif status == "stale":
            st.markdown(f'<p class="status-stale">● SYSTEM STALE</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="status-offline">● SYSTEM OFFLINE</p>', unsafe_allow_html=True)
        st.caption(health_status["message"])


def render_metrics(report: Optional[PerformanceReport]):
    """Render the key metrics row."""
    col1, col2, col3, col4, col5 = st.columns(5)

    if report is None:
        col1.metric("Total PnL", "$0.00")
        col2.metric("Win Rate", "0%")
        col3.metric("Total Trades", "0")
        col4.metric("Symbols", "0")
        col5.metric("Volume", "$0")
        return

    # Format PnL with color
    pnl = float(report.realized_pnl)
    pnl_str = f"${pnl:,.2f}"
    pnl_delta = "profit" if pnl >= 0 else "loss"

    col1.metric(
        "Realized PnL",
        pnl_str,
        delta=pnl_delta,
        delta_color="normal" if pnl >= 0 else "inverse",
    )

    col2.metric(
        "Win Rate",
        f"{report.win_rate:.1f}%",
        delta=f"{report.winning_trades}W / {report.losing_trades}L",
    )

    col3.metric(
        "Total Trades",
        f"{report.total_trades}",
        delta=f"{report.buy_trades} buys, {report.sell_trades} sells",
    )

    col4.metric(
        "Symbols Traded",
        f"{report.symbols_traded}",
    )

    col5.metric(
        "Total Volume",
        f"${float(report.total_volume):,.0f}",
    )


def render_pnl_chart(df: pd.DataFrame):
    """Render the cumulative PnL chart."""
    st.subheader("Cumulative P&L")

    if df is None or df.empty or "cumulative_pnl" not in df.columns:
        st.info("No trade data available for chart.")
        return

    # Create the chart
    fig = go.Figure()

    # Add area fill
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["cumulative_pnl"],
        mode="lines",
        name="Cumulative PnL",
        line=dict(color="#00d26a", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 210, 106, 0.1)",
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#666", opacity=0.5)

    # Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=350,
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            title="",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            title="P&L ($)",
            tickformat="$,.0f",
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_symbol_breakdown(report: Optional[PerformanceReport]):
    """Render per-symbol performance breakdown."""
    st.subheader("Performance by Symbol")

    if report is None or not report.per_symbol:
        st.info("No symbol data available.")
        return

    # Create DataFrame from per_symbol
    data = []
    for symbol, stats in report.per_symbol.items():
        data.append({
            "Symbol": symbol,
            "PnL": stats["realized_pnl"],
            "Buy Qty": stats["buy_qty"],
            "Sell Qty": stats["sell_qty"],
            "Avg Buy": stats["avg_buy_price"],
            "Avg Sell": stats["avg_sell_price"],
            "Status": "Closed" if stats["is_closed"] else "Open",
        })

    df = pd.DataFrame(data)
    df = df.sort_values("PnL", ascending=False)

    # Create horizontal bar chart
    colors = ["#00d26a" if x >= 0 else "#ff4757" for x in df["PnL"]]

    fig = go.Figure(go.Bar(
        x=df["PnL"],
        y=df["Symbol"],
        orientation="h",
        marker_color=colors,
        text=[f"${x:,.2f}" for x in df["PnL"]],
        textposition="outside",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=50, t=10, b=0),
        height=max(200, len(df) * 35),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            title="Realized P&L ($)",
            tickformat="$,.0f",
        ),
        yaxis=dict(
            showgrid=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_trade_table(df: pd.DataFrame):
    """Render the recent trades table."""
    st.subheader("Recent Trades")

    if df is None or df.empty:
        st.info("No trades recorded yet.")
        return

    # Format for display
    display_df = df[["timestamp", "symbol", "side", "qty", "price", "value", "status"]].copy()
    display_df = display_df.sort_values("timestamp", ascending=False)

    # Format columns
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["price"] = display_df["price"].apply(lambda x: f"${x:,.2f}")
    display_df["value"] = display_df["value"].apply(lambda x: f"${x:,.2f}")
    display_df["qty"] = display_df["qty"].apply(lambda x: f"{x:,.4f}".rstrip("0").rstrip("."))

    # Rename columns
    display_df.columns = ["Time", "Symbol", "Side", "Qty", "Price", "Value", "Status"]

    # Apply styling
    def style_side(val):
        if val == "BUY":
            return "color: #00d26a"
        elif val == "SELL":
            return "color: #ff4757"
        return ""

    styled = display_df.head(50).style.applymap(style_side, subset=["Side"])

    st.dataframe(styled, use_container_width=True, height=400)


def render_sidebar():
    """Render the sidebar with filters and settings."""
    st.sidebar.header("Settings")

    # Trade file path
    trades_file = st.sidebar.text_input(
        "Trades CSV Path",
        value="trades.csv",
        help="Path to the trades.csv audit file",
    )

    # Health file path
    health_file = st.sidebar.text_input(
        "Health File Path",
        value="/tmp/hybrid_trader.health",
        help="Path to the heartbeat file",
    )

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)

    st.sidebar.divider()

    # Manual refresh button
    if st.sidebar.button("Refresh Now", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    # Info section
    st.sidebar.caption("Hybrid Trader Dashboard v1.0")
    st.sidebar.caption("Data refreshes every 30 seconds")

    return trades_file, health_file, auto_refresh


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main dashboard application."""
    # Sidebar
    trades_file, health_file, auto_refresh = render_sidebar()

    # Check system health
    health_status = check_system_health(health_file)

    # Header
    render_header(health_status)

    st.divider()

    # Load data
    df = load_trades(trades_file)
    report = load_report(trades_file)

    # Metrics row
    render_metrics(report)

    st.divider()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Calculate cumulative PnL and render chart
        df_with_pnl = calculate_cumulative_pnl(df) if df is not None else None
        render_pnl_chart(df_with_pnl)

    with col2:
        render_symbol_breakdown(report)

    st.divider()

    # Trade table
    render_trade_table(df)

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(0.1)  # Small delay to prevent UI flicker
        st.empty()  # Trigger rerun mechanism


if __name__ == "__main__":
    main()
