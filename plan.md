# Implementation Roadmap

## Phase 1: Infrastructure & Data (Complete)
- [x] **Environment Stability:** Python 3.13 venv created, all dependencies (backtrader, pandas, alpaca, etc.) verified.
- [x] **Data Loader:** Basic `MarketDataLoader` implemented (supports yfinance).
- [x] **Data Normalizer:** Implement `HybridDataNormalizer` class to merge Crypto (24/7) and Stock (M-F) streams into a unified UTC index.
- [x] **Data Tests:** Verify weekend alignment logic via `pytest` (13 tests passing).

## Phase 2: Strategy Engine (Complete)
- [x] **Base Strategy:** `ReversalStrategy` abstract class with stop-loss (5%) and time-stop (5 days).
- [x] **Equity Strategy:** `EquityReversal` - weekly Friday rebalance, bottom decile ranking.
- [x] **Crypto Strategy:** `CryptoReversal` - ADX < 25 regime filter, dip buying.
- [x] **Feed Adapter:** `create_bt_feeds()` converts normalized DataFrame to backtrader feeds.
- [x] **Strategy Tests:** 15 tests passing (smoke, risk mgmt, integration), mypy clean.

## Phase 3: Execution Bridge (Complete)
- [x] **Data Models:** Typed DTOs (`AccountInfo`, `PositionInfo`, `OrderRequest`, `OrderResult`) with Decimal precision.
- [x] **Abstract Executor:** `ExecutorBase` ABC with full interface for order management.
- [x] **Alpaca Executor:** `AlpacaExecutor` wrapping alpaca-py with safety guards.
- [x] **Order Router:** `OrderRouter` converts strategy signals to orders with position sizing.
- [x] **Safety Mechanisms:** 5 layers (constructor guard, max order value, position limits, dry run, mocked tests).
- [x] **Execution Tests:** 24 tests passing (all mocked, no real orders), mypy clean.

## Phase 4: Integration & Main Loop (Complete)
- [x] **Configuration:** `TradingConfig` dataclass with python-dotenv loading, validation, and live trading warning banner.
- [x] **Main Entry Point:** `src/main.py` with CLI (`--dry-run`, `--once`), trading cycle loop, and error resilience.
- [x] **Signal Extraction:** Added `last_signals` property to `ReversalStrategy` for signal capture after cerebro.run().
- [x] **Price Fetcher:** Caching price fetcher for OrderRouter position sizing.
- [x] **Integration Tests:** 20 tests covering config, signal extraction, pipeline flow (72 total tests passing).
- [x] **Type Safety:** mypy clean on all Phase 4 files.

## Phase 5: Production Hardening (Complete)
- [x] **Structured Logging:** `src/utils/logger.py` with JSON formatter and `TradeLogger` CSV audit trail.
- [x] **Health Monitoring:** `src/health.py` with `HealthMonitor` class (heartbeat file, API connectivity check).
- [x] **Performance Reporting:** `src/reporting.py` standalone script (PnL, Win Rate, per-symbol breakdown).
- [x] **Main Loop Integration:** Updated `src/main.py` with `--json-logs`, health checks, trade audit logging.
- [x] **Documentation:** Comprehensive `README.md` with deployment instructions (systemd, Docker, K8s).
- [x] **Type Safety:** mypy clean on all Phase 5 files (72 tests passing).

## Phase 6: Deployment & Visualization (Complete)
- [x] **Dashboard:** `src/dashboard.py` - Professional Streamlit quant terminal with PnL chart, symbol breakdown, trade table.
- [x] **Deploy Script:** `deploy.sh` - Git pull, venv setup, tests, systemd restart with `--no-restart` and `--test-only` flags.
- [x] **Systemd Service:** `systemd/hybrid_trader.service` - Production service file with security hardening.
- [x] **Dependencies:** Added `streamlit` and `plotly` to requirements.txt.
- [x] **Documentation:** Updated README with Dashboard section and deploy script usage.

## Phase 7: Paper Trading Validation (Future)
- [ ] Run system in paper mode for 1+ week to validate end-to-end behavior.
- [ ] Analyze trade logs and performance reports.
- [ ] Tune strategy parameters based on real market conditions.
- [ ] Implement Prometheus/Grafana metrics export.
- [ ] Add WebSocket real-time price streaming for lower latency.
