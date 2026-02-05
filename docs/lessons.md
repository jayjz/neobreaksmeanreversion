# Project Lessons & Learned Patterns

## Environment & Architecture (Confirmed 2026-02-05)
- **Python 3.13 is Viable:** We are running successfully on Python 3.13.11 with standard pip wheels. No downgrades needed.
- **Backtrader Pivot:** We have officially dropped `vectorbt` in favor of `backtrader` for event-driven stability with hybrid assets.
- **Library Nuances:**
  - `alpaca-py`: Package name is `alpaca-py`, but import usage is `import alpaca` (NOT `import alpaca_py`).
  - `pandas-ta`: We are on the Beta branch. Use `df.ta.ticker('SPY')` for data loading tests, not legacy examples.
  - `venv`: Always run inside the dedicated `venv`.

## Data Strategy (Phase 1)
- **Hybrid Alignment:** Crypto trades 24/7; Stocks trade M-F.
- **Normalization Rule:** We must **forward-fill (ffill)** Stock data over weekends/holidays. The equity curve must remain "flat" on Saturday/Sunday while Crypto prices move. Never drop Crypto weekends to match Stocks.
- **Unified Index:** Use timezone-aware UTC daily index for all feeds to prevent alignment bugs.

## Strategy Engine (Phase 2 - 2026-02-05)
- **Indicator Choice:** Prefer native backtrader indicators (e.g., `bt.indicators.ADX`) over pandas-ta equivalents for perfect bar-by-bar alignment and automatic handling.
- **Risk Management:** Manual stop-loss/time-stop tracking in `next()` is more testable and reliable than bracket orders or built-in stops.
- **Type Checking:** Use `# type: ignore[assignment]` for backtrader `params` inheritance — mypy limitation, but runtime behavior is correct.
- **Subclass Design:** Separate strategy subclasses per asset class (EquityReversal, CryptoReversal) for clean divergence in logic (weekly ranking vs. ADX gating).
- **Feed Adapter:** Always convert normalized MultiIndex DataFrame → individual PandasData feeds with explicit column mapping and ticker naming.
- **Rebalance Timing:** Explicit `self.datas[0].datetime.date(0).weekday() == 4` for Friday checks — simple and reliable.

## Execution Bridge (Phase 3 - 2026-02-05)
- **Safety First:** Multiple layers of protection against accidental live trading: constructor guard (`paper=False` requires `live_trading_enabled=True`), max order value, position limits, dry run mode.
- **Alpaca-py Types:** API returns union types (e.g., `Order | dict[str, Any]`). Use `# type: ignore[arg-type]` for mypy compatibility — runtime behavior is correct.
- **Decimal Precision:** All monetary values use `Decimal` (not `float`) to prevent floating-point errors in financial calculations.
- **Frozen Dataclasses:** DTOs (`AccountInfo`, `PositionInfo`, `OrderResult`) are frozen for immutability and thread safety.
- **Crypto vs Equity:** Different TimeInForce values — crypto only supports `GTC`/`IOC`, equities use `DAY`. Handle in `_build_alpaca_request()`.
- **Mocked Tests:** All execution tests use `unittest.mock` to prevent real API calls. Never test against live Alpaca endpoints.
- **Order Router Design:** Synchronous design matches backtrader's model. Async can be added later if needed for high-frequency use cases.

## Integration & Main Loop (Phase 4 - 2026-02-05)
- **Signal Extraction from Backtrader:** Strategy's `generate_signals()` is called bar-by-bar. Store signals in `_last_signals` and expose via `last_signals` property for retrieval after `cerebro.run()`.
- **Config Testing with dotenv:** When testing config loading, `patch.dict(os.environ, clear=True)` clears env, but `from dotenv import load_dotenv` creates a local reference. Must use `patch.object(config, "load_dotenv")` AFTER `importlib.reload(config)` to properly mock.
- **Price Fetcher Pattern:** OrderRouter needs current prices for position sizing. Create a caching price fetcher function that wraps MarketDataLoader to avoid redundant API calls.
- **Live Trading Warning:** Print high-visibility banner and sleep 5 seconds when `TRADING_MODE=LIVE` to give operators a chance to abort.
- **CLI Design:** Use `argparse` with `--dry-run` and `--once` flags. Dry-run passes through to `router.execute_signals(signals, dry_run=True)`.
- **Error Resilience:** Main loop wraps `run_cycle()` in try/except — log errors but continue to next cycle. Trading systems must not crash on transient failures.
- **Module Reload in Tests:** When testing modules that read env vars at load time, use `importlib.reload()` inside the patched environment context to ensure fresh state.

## Production Hardening (Phase 5 - 2026-02-05)
- **JSON Logging for Observability:** Use a custom `JSONFormatter` class that outputs structured logs compatible with Splunk/Datadog/ELK. Include `timestamp`, `level`, `logger`, `message`, plus any extra fields.
- **Trade Audit Trail:** Always log executed trades to a CSV file (`TradeLogger`). This provides compliance records and input for performance analysis.
- **Heartbeat File Pattern:** Touch a file (e.g., `/tmp/hybrid_trader.health`) each cycle. External monitoring tools (Monit, K8s liveness probes) can watch mtime to detect hung processes.
- **API Health Check:** Use a lightweight API call (account info) to verify connectivity. Run on startup and periodically during operation.
- **FIFO PnL Matching:** When calculating realized PnL, match buys to sells in FIFO order. Track open positions as a queue per symbol.
- **Type Safety with sum():** `sum(generator)` returns 0 (int) for empty iterables. Always provide a start value: `sum(gen, Decimal("0"))` to maintain Decimal type.
- **Container-Friendly Logging:** Write logs to stdout (not files) for Docker/K8s. Use `--json-logs` flag for production deployments.
- **Deployment Options:** Document multiple deployment patterns (cron for simple, systemd for production, Docker/K8s for cloud) in README.

## Deployment & Visualization (Phase 6 - 2026-02-05)
- **Streamlit Caching:** Use `@st.cache_data(ttl=30)` for data loaders. TTL prevents stale data while reducing API calls.
- **Plotly Dark Theme:** Use `template="plotly_dark"` with `paper_bgcolor="rgba(0,0,0,0)"` for transparent backgrounds that blend with Streamlit's dark mode.
- **Status Indicators:** Heartbeat file mtime comparison is simple and portable. Calculate age as `datetime.now(timezone.utc) - mtime` for consistent behavior.
- **Cumulative PnL Calculation:** Track positions per symbol with avg price. Only realize PnL on SELL (simplified but reasonable for display purposes).
- **Deploy Script Design:** Use `set -euo pipefail` for bash safety. Separate steps into functions for clarity. Support flags like `--no-restart` for CI/CD flexibility.
- **Systemd Security:** Use `ProtectSystem=strict`, `ProtectHome=read-only`, `NoNewPrivileges=true` for security hardening. Allow only necessary write paths.
- **Service Restart Policy:** `RestartSec=60` with `StartLimitBurst=3` in `StartLimitInterval=300` prevents restart loops while ensuring recovery from transient failures.
