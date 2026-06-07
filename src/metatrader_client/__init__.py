"""
MetaTrader MCP Client package.

This package provides a modular interface for communicating with the MetaTrader 5 terminal.
"""

# MT5Client / MT5Order pull in the MetaTrader5 package, which is Windows-only.
# Import them gracefully so this package still imports on Linux / macOS when a
# hosted provider is used instead of a local terminal (TICKERALL_API_KEY). When
# MetaTrader5 is installed (the local-MT5 path), this behaves exactly as before.
try:
    from .client import MT5Client
    from .client_order import MT5Order
except ImportError:
    MT5Client = None  # type: ignore[assignment,misc]
    MT5Order = None  # type: ignore[assignment,misc]

from .exceptions import (
    MT5ClientError, 
    ConnectionError, 
    OrderError, 
    MarketError,
    AccountError,
    HistoryError
)

__all__ = [
    
    "MT5Client",
    "MT5Order",

    "MT5ClientError",
    "ConnectionError",
    "OrderError",
    "MarketError",
    "AccountError",
    "HistoryError",
]
