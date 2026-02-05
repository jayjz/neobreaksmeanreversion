# Hybrid Trader

A production-grade algorithmic trading system for hybrid assets (equities + crypto) using mean-reversion strategies.

## Features

- **Hybrid Asset Support**: Trades both S&P 500 equities and crypto (BTC, ETH)
- **Mean Reversion Strategies**:
  - Equity: Weekly bottom-decile rebalancing
  - Crypto: ADX-gated volatility regime filter
- **Risk Management**: 5% hard stop-loss, 5-day time stops
- **Safety First**: Multiple layers of protection against accidental live trading
- **Production Ready**: JSON logging, health monitoring, trade audit trails

## Quick Start

### 1. Prerequisites

- Python 3.13+
- Alpaca trading account (paper or live)

### 2. Installation

```bash
# Clone and setup
git clone <repo-url>
cd hybrid_trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy example config
cp .env.example .env

# Edit with your Alpaca credentials
nano .env
```

Required environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `ALPACA_API_KEY` | Your Alpaca API key | `PKXXXXXXXXXX` |
| `ALPACA_SECRET_KEY` | Your Alpaca secret key | `xxxxxxxxxxxxxxxx` |
| `TRADING_MODE` | `PAPER` or `LIVE` | `PAPER` |
| `LIVE_TRADING_ENABLED` | Must be `true` for live trading | `false` |

### 4. Run

```bash
# Paper trading (safe)
python -m src.main --dry-run --once

# Paper trading loop
python -m src.main

# With JSON logs (for Splunk/Datadog)
python -m src.main --json-logs

# Single cycle then exit
python -m src.main --once
```

## Architecture

```
src/
├── main.py              # Entry point, trading loop
├── config.py            # Environment configuration
├── health.py            # Health monitoring
├── reporting.py         # Performance analysis
├── data/
│   ├── loader.py        # Market data fetching (yfinance)
│   └── normalizer.py    # Hybrid data alignment
├── strategies/
│   ├── base.py          # ReversalStrategy base class
│   ├── equity.py        # EquityReversal (weekly rebalance)
│   ├── crypto.py        # CryptoReversal (ADX filter)
│   └── feed_adapter.py  # Backtrader data feeds
├── execution/
│   ├── alpaca.py        # Alpaca API wrapper
│   ├── router.py        # Signal → Order conversion
│   └── models.py        # DTOs (OrderRequest, OrderResult)
└── utils/
    └── logger.py        # JSON logging, trade audit
```

## Dashboard

A professional quant terminal built with Streamlit for real-time performance monitoring.

### Launch Dashboard

```bash
# Basic launch
streamlit run src/dashboard.py

# With custom data paths
streamlit run src/dashboard.py -- --trades trades.csv --health /tmp/hybrid_trader.health
```

### Features

- **Real-time Metrics**: PnL, Win Rate, Trade Count, Volume
- **Cumulative P&L Chart**: Interactive Plotly visualization
- **Per-Symbol Breakdown**: Horizontal bar chart of gains/losses
- **Recent Trades Table**: Color-coded buy/sell with timestamps
- **System Status**: Live/Stale/Offline indicator based on heartbeat file
- **Auto-refresh**: Data updates every 30 seconds

### Screenshot

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  HYBRID TRADER                                        ● SYSTEM ONLINE     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Realized PnL    Win Rate    Total Trades    Symbols    Volume            ║
║  $1,234.56       65.0%       42              8          $125,000          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  [Cumulative P&L Chart]              │  [Per-Symbol Breakdown]            ║
║  ████████████████████▓               │  AAPL  ████████  $450              ║
║  █████████████████▓                  │  MSFT  ██████    $320              ║
║  ████████████▓                       │  GOOGL ████      $180              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Recent Trades                                                            ║
║  Time           Symbol   Side   Qty      Price      Value      Status     ║
║  2026-02-05     AAPL     BUY    10       $150.50    $1,505     FILLED     ║
║  2026-02-05     MSFT     SELL   5        $420.00    $2,100     FILLED     ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Command Reference

### Main Trading Loop

```bash
# Basic usage
python -m src.main [OPTIONS]

Options:
  --dry-run      Generate signals but don't place orders
  --once         Run single cycle and exit
  --json-logs    Output JSON formatted logs
  --log-file     Write logs to file (in addition to stdout)
  --trade-log    Path to trade audit CSV (default: trades.csv)
```

### Health Check

```bash
# Standalone health check (for K8s probes, monitoring)
python -m src.health
```

Returns exit code 0 if healthy, 1 if unhealthy.

### Performance Report

```bash
# Generate report from trade log
python -m src.reporting

# JSON output
python -m src.reporting --json

# Custom file
python -m src.reporting --file /path/to/trades.csv
```

## Production Deployment

### Automated Deployment Script

Use the included `deploy.sh` for streamlined deployments:

```bash
# Full deployment (git pull, deps, tests, service restart)
./deploy.sh

# Deploy without restarting the service
./deploy.sh --no-restart

# Only run tests (CI/CD validation)
./deploy.sh --test-only
```

The script will:
1. Pull latest code from git (stashes uncommitted changes)
2. Create/update Python virtual environment
3. Run full test suite (aborts on failure)
4. Run type checks (non-blocking warnings)
5. Restart systemd service

### Cron Job (Simple)

```bash
# Run every hour at minute 30
30 * * * * cd /path/to/hybrid_trader && venv/bin/python -m src.main --once >> /var/log/trader.log 2>&1
```

### Systemd Service (Recommended)

Create `/etc/systemd/system/hybrid-trader.service`:

```ini
[Unit]
Description=Hybrid Trader
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/opt/hybrid_trader
Environment=PATH=/opt/hybrid_trader/venv/bin
ExecStart=/opt/hybrid_trader/venv/bin/python -m src.main --json-logs
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable hybrid-trader
sudo systemctl start hybrid-trader
```

### Docker

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Health check
HEALTHCHECK --interval=60s --timeout=10s \
  CMD python -m src.health || exit 1

CMD ["python", "-m", "src.main", "--json-logs"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hybrid-trader
spec:
  replicas: 1  # IMPORTANT: Only run one instance
  template:
    spec:
      containers:
      - name: trader
        image: hybrid-trader:latest
        envFrom:
        - secretRef:
            name: alpaca-credentials
        livenessProbe:
          exec:
            command: ["python", "-m", "src.health"]
          initialDelaySeconds: 30
          periodSeconds: 60
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Monitoring

### Health File

The trader updates `/tmp/hybrid_trader.health` every cycle. Monitor with:

```bash
# Monit
check file hybrid_trader_health with path /tmp/hybrid_trader.health
  if timestamp > 10 minutes then alert

# Simple script
if [ $(find /tmp/hybrid_trader.health -mmin +10 2>/dev/null) ]; then
  echo "ALERT: Trader may be stuck"
fi
```

### Logs

JSON logs are compatible with:
- Splunk
- Datadog
- ELK Stack
- Grafana Loki

Example log entry:
```json
{"timestamp": "2026-02-05T10:30:00Z", "level": "INFO", "logger": "src.main", "message": "Cycle completed", "cycle": 42, "signals_count": 3}
```

### Trade Audit Trail

All executed trades are logged to `trades.csv`:

```csv
timestamp,symbol,side,qty,price,order_id,status
2026-02-05T10:30:00Z,AAPL,BUY,10,150.50,order-123,FILLED
```

## Safety Mechanisms

1. **Paper Mode Default**: `TRADING_MODE=PAPER` unless explicitly set to `LIVE`
2. **Explicit Live Flag**: Live trading requires `LIVE_TRADING_ENABLED=true`
3. **Startup Warning**: 5-second delay with warning banner when in LIVE mode
4. **Max Order Value**: Configurable cap on individual order size
5. **Dry Run Mode**: `--dry-run` flag for testing without orders
6. **API Health Check**: Validates connectivity before starting

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_integration.py -v

# Type checking
mypy src/ --ignore-missing-imports
```

## Troubleshooting

### "API health check failed on startup"

- Verify `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are correct
- Check if using paper keys with `TRADING_MODE=PAPER`
- Ensure network connectivity to Alpaca API

### "No signals generated"

- Strategy only generates signals on specific conditions:
  - EquityReversal: Friday only (weekly rebalance)
  - CryptoReversal: ADX < 25 (ranging market)
- Check if you have enough historical data (`LOOKBACK_DAYS`)

### "SafetyError: Live trading requires explicit flag"

- This is intentional! To enable live trading:
  ```
  TRADING_MODE=LIVE
  LIVE_TRADING_ENABLED=true
  ```

## Development

### Project Structure

```
hybrid_trader/
├── src/           # Source code
├── tests/         # Test suite
├── docs/          # Documentation
├── .env.example   # Config template
├── requirements.txt
└── README.md
```

### Adding a New Strategy

1. Subclass `ReversalStrategy` in `src/strategies/`
2. Implement `generate_signals()` and `get_position_size()`
3. Register in `src/strategies/__init__.py`
4. Add tests in `tests/test_strategies.py`

## License

MIT

## Disclaimer

This software is for educational purposes. Algorithmic trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.
