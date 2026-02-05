"""
Structured logging and trade audit trail for Hybrid Trader.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional, TextIO

# Define the standard schema for trades.csv
TRADE_COLUMNS = ["timestamp", "symbol", "side", "qty", "price", "order_id", "status", "value"]

class JsonFormatter(logging.Formatter):
    """Format logs as JSON for Splunk/Datadog."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_obj.update(record.extra)  # type: ignore
        # Merge extra fields passed via logging.info(..., extra={...})
        if record.__dict__.get("extra"):
             log_obj.update(record.__dict__["extra"])
        
        return json.dumps(log_obj)

def setup_logging(level: int = logging.INFO, json_format: bool = False, log_file: Optional[str] = None) -> None:
    """Configure root logger."""
    root = logging.getLogger()
    root.setLevel(level)
    
    # Clear existing handlers
    root.handlers = []
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    if json_format:
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    root.addHandler(console)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
        root.addHandler(file_handler)

class TradeLogger:
    """Logs executed trades to CSV."""
    def __init__(self, filepath: str = "trades.csv"):
        self.filepath = filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create file with header if missing."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(TRADE_COLUMNS)
                
    def log_order(self, order: Any, side: str, status: str = "FILLED") -> None:
        """Append trade to CSV."""
        # Calculate value safely
        qty = float(order.qty) if order.qty else 0.0
        price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
        value = qty * price

        row = [
            datetime.now(timezone.utc).isoformat(),
            order.symbol,
            side,
            order.qty,
            order.filled_avg_price,
            order.id,
            status,
            f"{value:.2f}"
        ]
        
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
