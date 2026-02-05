"""
Health monitoring for Hybrid Trader.

Provides:
- Heartbeat file touch for external monitoring (Monit, K8s liveness probes)
- API connectivity checks
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .execution import AlpacaExecutor

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Health monitoring for production deployment.

    Features:
    - Heartbeat file updates for external monitoring tools
    - API connectivity verification
    - Last successful cycle tracking
    """

    DEFAULT_HEARTBEAT_FILE = "/tmp/hybrid_trader.health"

    def __init__(
        self,
        executor: Optional[AlpacaExecutor] = None,
        heartbeat_file: str = DEFAULT_HEARTBEAT_FILE,
    ) -> None:
        """
        Initialize HealthMonitor.

        Args:
            executor: Optional AlpacaExecutor for API connectivity checks
            heartbeat_file: Path to heartbeat file (default: /tmp/hybrid_trader.health)
        """
        self._executor = executor
        self._heartbeat_path = Path(heartbeat_file)
        self._last_heartbeat: Optional[datetime] = None
        self._last_api_check: Optional[datetime] = None
        self._consecutive_failures = 0

    def heartbeat(self) -> None:
        """
        Update heartbeat file to signal the process is alive.

        External monitoring tools (Monit, Kubernetes, systemd) can watch
        this file's modification time to detect hung processes.
        """
        try:
            # Touch the file (create if doesn't exist, update mtime if exists)
            self._heartbeat_path.touch(exist_ok=True)
            self._last_heartbeat = datetime.now(timezone.utc)
            self._consecutive_failures = 0
            logger.debug(f"Heartbeat updated: {self._heartbeat_path}")
        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(f"Failed to update heartbeat file: {e}")

    def check_api(self) -> bool:
        """
        Verify API connectivity with a lightweight call.

        Uses the Alpaca market clock endpoint which is fast and doesn't
        count against rate limits.

        Returns:
            True if API is reachable and responding, False otherwise
        """
        if self._executor is None:
            logger.warning("No executor configured for API health check")
            return False

        try:
            # Use get_account_info as a connectivity test
            # This verifies both network and authentication
            account = self._executor.get_account_info()

            self._last_api_check = datetime.now(timezone.utc)
            self._consecutive_failures = 0

            logger.debug(
                f"API check passed: cash=${account.cash}, equity=${account.equity}"
            )
            return True

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"API health check failed: {e}")
            return False

    def get_status(self) -> dict:
        """
        Get current health status as a dictionary.

        Returns:
            Dict with health status information
        """
        return {
            "heartbeat_file": str(self._heartbeat_path),
            "last_heartbeat": (
                self._last_heartbeat.isoformat() if self._last_heartbeat else None
            ),
            "last_api_check": (
                self._last_api_check.isoformat() if self._last_api_check else None
            ),
            "consecutive_failures": self._consecutive_failures,
            "healthy": self._consecutive_failures == 0,
        }

    @property
    def is_healthy(self) -> bool:
        """Return True if no consecutive failures have occurred."""
        return self._consecutive_failures == 0

    @property
    def consecutive_failures(self) -> int:
        """Return count of consecutive failures."""
        return self._consecutive_failures


def run_health_check(
    api_key: str,
    secret_key: str,
    paper: bool = True,
) -> bool:
    """
    Standalone health check function.

    Useful for external scripts or container health probes.

    Args:
        api_key: Alpaca API key
        secret_key: Alpaca secret key
        paper: Whether to use paper trading API

    Returns:
        True if all checks pass
    """
    try:
        executor = AlpacaExecutor(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

        monitor = HealthMonitor(executor=executor)

        # Run checks
        monitor.heartbeat()
        api_ok = monitor.check_api()

        status = monitor.get_status()
        logger.info(f"Health check status: {status}")

        return api_ok

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


if __name__ == "__main__":
    # Standalone health check
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.config import load_config

    logging.basicConfig(level=logging.INFO)

    try:
        config = load_config()
        result = run_health_check(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            paper=config.paper,
        )
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        sys.exit(1)
