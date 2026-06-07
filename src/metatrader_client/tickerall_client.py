"""
Optional hosted-API client for the MetaTrader MCP server.

This is an **additive, opt-in** drop-in alternative to ``MT5Client`` that talks
to a hosted MetaTrader API instead of a local MT5 terminal. It mirrors the same
public surface (``client.account`` / ``client.market`` / ``client.order`` /
``client.history`` plus ``connect`` / ``disconnect`` / ``is_connected``), so the
MCP tools — and the OpenAPI / quote servers — work through it unchanged.

It is selected purely by the ``TICKERALL_API_KEY`` environment variable (see
``metatrader_mcp.utils.init``). When that variable is unset, this module is never
imported and the local-MT5 path behaves exactly as before. So:

- No ``TICKERALL_API_KEY``  → local MT5 terminal (Windows), unchanged.
- ``TICKERALL_API_KEY`` set → hosted API: runs on Linux / macOS / anywhere, no
  terminal to install or babysit.

The hosted ``tickerall`` package is an optional dependency; it's imported lazily
inside :meth:`TickerAllClient.connect` so installing it is only required when you
actually opt in.
"""
from __future__ import annotations

import fnmatch
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Minutes per MT5 timeframe code — used to translate a "latest N candles" request
# into the hosted API's hours-of-lookback parameter.
_TF_MINUTES = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6, "M10": 10, "M12": 12,
    "M15": 15, "M20": 20, "M30": 30, "H1": 60, "H2": 120, "H3": 180, "H4": 240,
    "H6": 360, "H8": 480, "H12": 720, "D1": 1440, "W1": 10080, "MN1": 43200,
}


def _tf(timeframe: str) -> str:
    tf = (timeframe or "").upper()
    if tf not in _TF_MINUTES:
        raise ValueError(f"Invalid timeframe: '{timeframe}'")
    return tf


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True)
    return ts.to_pydatetime()


def _inclusive_to(to_date):
    """A bare YYYY-MM-DD `to` means 'through the END of that day' (MT5 history
    semantics). The hosted API parses a date-only string to midnight, so a
    same-day / from==to query filters closeTime <= 00:00:00 and returns EMPTY —
    silently hiding that day's trades (audit §1). Bump a date-only `to` to
    end-of-day so the whole day is included. Times-of-day are passed through
    untouched. Returns a string (history passes strings to the SDK)."""
    if not to_date:
        return to_date
    s = str(to_date).strip()
    if len(s) == 10 and s.count("-") == 2 and "T" not in s and ":" not in s:
        return s + "T23:59:59.999999"
    return to_date


# MT5 ENUM_SYMBOL_TRADE_MODE — surface a human label alongside the raw int so a
# caller doesn't have to memorise the enum.
_TRADE_MODE_DESC = {0: "DISABLED", 1: "LONGONLY", 2: "SHORTONLY", 3: "CLOSEONLY", 4: "FULL"}


def _tick_freshness(tick) -> tuple:
    """(age_seconds, stale) for a cached tick — stale past 60s so a bot can
    reject a frozen weekend / closed-market quote (audit §14). Shared by both
    get_symbol_price and the bid/ask embedded in get_symbol_info."""
    ts = _parse_date(getattr(tick, "timestamp", None))
    if ts is None:
        return None, False
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    return round(age, 1), age > 60.0


# ── DataFrame builders (match the local-MT5 client's column shapes) ───────────

def _positions_to_df(positions) -> pd.DataFrame:
    if not positions:
        return pd.DataFrame()
    rows = [{
        "ticket": p.ticket,
        "time": p.open_time,
        "type": p.side,                 # 'BUY' / 'SELL'
        "magic": p.magic,
        "volume": p.volume,
        "price_open": p.entry_price,
        "sl": p.stop_loss,
        "tp": p.take_profit,
        "price_current": p.current_price,
        "swap": p.swap,
        "commission": p.commission,
        "profit": p.profit,
        "symbol": p.symbol,
        "comment": p.comment,
    } for p in positions]
    return pd.DataFrame(rows)


def _pending_to_df(orders) -> pd.DataFrame:
    if not orders:
        return pd.DataFrame()
    rows = [{
        "ticket": o.ticket,
        "time_setup": o.set_time,
        "type": o.type,                 # BUY_LIMIT / SELL_STOP / ...
        "symbol": o.symbol,
        "volume": o.volume,
        "price_open": o.price,
        "sl": o.stop_loss,
        "tp": o.take_profit,
        "expiration": o.expiration_time,
    } for o in orders]
    return pd.DataFrame(rows)


def _candles_to_df(candles) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "time": c.timestamp,
        "open": c.open,
        "high": c.high,
        "low": c.low,
        "close": c.close,
        # Real per-bar tick count + spread (ask−bid) when the SDK/API supplies
        # them; getattr keeps this working against an older SDK that doesn't.
        "tick_volume": int(getattr(c, "tick_volume", 0) or 0),
        "spread": float(getattr(c, "spread", 0.0) or 0.0),
    } for c in candles])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time", ascending=False).reset_index(drop=True)


def _history_to_df(trades) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = [{
        "ticket": t.ticket,
        "symbol": t.symbol,
        "type": t.side,
        "volume": t.volume,
        "open_price": t.open_price,
        "close_price": t.close_price,
        "open_time": t.open_time,
        "close_time": t.close_time,
        # Coerce missing numerics to 0.0 (audit §2): the exit-deal leg of a
        # round-trip can carry a blank profit, which a CSV consumer parsing
        # `profit` as float would hit as NaN / an exception. (The degenerate
        # zero-duration legs themselves are a server-side pairing concern.)
        "profit": t.profit if t.profit is not None else 0.0,
        "swap": t.swap if t.swap is not None else 0.0,
        "commission": t.commission if t.commission is not None else 0.0,
    } for t in trades]
    # The hosted history can surface both constituent deals of a round-trip
    # (entry + exit) as separate records that project to byte-identical rows.
    # Drop only fully-identical rows — conservative, never removes a genuinely
    # distinct trade.
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _orders_to_df(orders) -> pd.DataFrame:
    """One row per FILLED order (distinct from the deal/round-trip view)."""
    if not orders:
        return pd.DataFrame()
    return pd.DataFrame([{
        "order_ticket": o.order_ticket,
        "symbol": o.symbol,
        "type": o.side,
        "volume": o.volume,
        "price": o.price,
        "time": o.time,
        "position_id": o.position_id,
        "state": o.state,
        "deal_count": o.deal_count,
    } for o in orders])


def _ok(message: str, data: Any = None) -> Dict[str, Any]:
    return {"error": False, "message": message, "data": data}


def _err(message: str) -> Dict[str, Any]:
    return {"error": True, "message": message, "data": None}


# ── Sub-adapters ──────────────────────────────────────────────────────────────

class _Account:
    def __init__(self, owner: "TickerAllClient"):
        self._o = owner

    def _info(self):
        return self._o._sdk.accounts.get(self._o._account_id).account

    def get_account_info(self) -> Dict[str, Any]:
        return self.get_trade_statistics()

    def get_trade_statistics(self) -> Dict[str, Any]:
        info = self._info()
        if info is None:
            return {"error": True, "message": "Account offline — connection is warming, retry shortly."}
        equity = info.equity if info.equity is not None else info.balance
        profit = (equity - info.balance) if equity is not None else 0.0
        return {
            "name": info.name,
            "server": self._o._server,
            "account_type": info.account_type,
            "currency": info.currency,
            "leverage": info.leverage,
            "balance": info.balance,
            "equity": equity,
            "profit": profit,
            "margin": info.margin,
            "free_margin": info.free_margin,
            "margin_level": info.margin_level,
        }

    # Full MT5Account surface (each derived from the account snapshot) so this is
    # a faithful drop-in, not just the subset the MCP/HTTP tools currently call.
    def get_balance(self) -> float:
        i = self._info(); return float(i.balance) if i else 0.0

    def get_equity(self) -> float:
        i = self._info()
        if not i:
            return 0.0
        return float(i.equity if i.equity is not None else i.balance)

    def get_margin(self) -> float:
        i = self._info(); return float(i.margin or 0.0) if i else 0.0

    def get_free_margin(self) -> float:
        i = self._info()
        if not i:
            return 0.0
        return float(i.free_margin if i.free_margin is not None else i.balance)

    def get_margin_level(self) -> float:
        i = self._info(); return float(i.margin_level or 0.0) if i else 0.0

    def get_currency(self) -> str:
        i = self._info(); return (i.currency or "") if i else ""

    def get_leverage(self) -> int:
        i = self._info(); return int(i.leverage) if i else 0

    def get_account_type(self) -> str:
        i = self._info(); return (i.account_type or "") if i else ""

    def is_trade_allowed(self) -> bool:
        return self._o._connected

    def check_margin_level(self, min_level: float = 100.0) -> bool:
        ml = self.get_margin_level()
        return ml >= min_level if ml else True


class _Market:
    def __init__(self, owner: "TickerAllClient"):
        self._o = owner

    def get_symbols(self, group: Optional[str] = None) -> List[str]:
        symbols = self._o._sdk.accounts.symbols(self._o._account_id)
        if group:
            symbols = [s for s in symbols if fnmatch.fnmatch(s, group)]
        return symbols

    def get_symbol_price(self, symbol_name: str) -> Dict[str, Any]:
        # Distinguish an UNKNOWN symbol from a known-but-closed market (audit §7)
        # so a bot doesn't retry forever on a typo.
        if not self._o._symbol_known(symbol_name):
            raise RuntimeError(
                f"Symbol '{symbol_name}' is not offered on this account "
                f"(unknown symbol — not a closed market)."
            )
        tick = self._o._latest_tick(symbol_name)
        age, stale = _tick_freshness(tick)
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.bid,
            "volume": 0,
            "time": _parse_date(tick.timestamp) or datetime.now(timezone.utc),
            # Freshness guard (audit §14): weekend / closed-market quotes are
            # served with an honest timestamp but a frozen, non-tradeable ask
            # (e.g. EURUSD with a ~1200-pip Friday-close spread). Surface the
            # quote age + a `stale` flag so a 24/5 bot can refuse to trade on a
            # frozen quote instead of buying far above the real bid.
            "age_seconds": age,
            "stale": stale,
        }

    def get_candles_latest(self, symbol_name: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        tf = _tf(timeframe)
        # Tell an UNKNOWN symbol (typo / unsupported) apart from a known symbol
        # with no bars in the window (audit §7) — otherwise a typo'd symbol
        # silently returns an empty frame instead of erroring, exactly like
        # get_symbol_price guards.
        if not self._o._symbol_known(symbol_name):
            raise RuntimeError(
                f"Symbol '{symbol_name}' is not offered on this account "
                f"(unknown symbol — not a closed market)."
            )
        # Translate "N bars" into a generous hours look-back, then take the most
        # recent N. The 3x buffer + a floor absorb weekend/holiday gaps and reach
        # far enough back to trigger the deep-history fetch, so a request still
        # yields ~N bars even when the live resident window is shallow.
        hours = max(72, int((count * _TF_MINUTES[tf] / 60) * 3) + 1)
        candles = self._o._sdk.candles.get(self._o._account_id, symbol=symbol_name, hours=hours, timeframe=tf)
        return _candles_to_df(candles).head(int(count))

    def get_candles_by_date(self, symbol_name: str, timeframe: str,
                            from_date: Optional[str] = None, to_date: Optional[str] = None) -> pd.DataFrame:
        tf = _tf(timeframe)
        # Unknown symbol -> error (not a silent empty frame), same guard as
        # get_candles_latest / get_symbol_price (audit §7).
        if not self._o._symbol_known(symbol_name):
            raise RuntimeError(
                f"Symbol '{symbol_name}' is not offered on this account "
                f"(unknown symbol — not a closed market)."
            )
        frm = _parse_date(from_date)
        to = _parse_date(_inclusive_to(to_date))  # date-only `to` = end of that day (audit §1)
        now = datetime.now(timezone.utc)
        span_start = frm or (now - pd.Timedelta(days=7).to_pytimedelta())
        hours = max(1, int((now - span_start).total_seconds() / 3600) + 1)
        candles = self._o._sdk.candles.get(self._o._account_id, symbol=symbol_name, hours=hours, timeframe=tf)
        df = _candles_to_df(candles)
        if df.empty:
            return df
        if frm is not None:
            df = df[df["time"] >= pd.Timestamp(frm)]
        if to is not None:
            df = df[df["time"] <= pd.Timestamp(to)]
        return df.reset_index(drop=True)

    def get_symbol_info(self, symbol_name: str) -> Dict[str, Any]:
        """Symbol spec (digits/point/contract/lot limits/trade mode) plus a live
        bid/ask, keyed with MT5-style names. Raises if the symbol is unknown to
        the account (no spec and no obtainable tick)."""
        spec = self._o._symbol_spec(symbol_name)
        tick = None
        try:
            # Match get_symbol_price's window (4s) so a known-but-closed market's
            # frozen snapshot tick is surfaced here too, not dropped.
            tick = self._o._latest_tick(symbol_name, timeout=4.0)
        except Exception:  # noqa: BLE001
            tick = None
        if spec is None and tick is None:
            raise RuntimeError(f"Symbol '{symbol_name}' not found on this account.")
        info: Dict[str, Any] = {"name": symbol_name}
        if spec is not None:
            info.update({
                "digits": spec.digits,
                "point": spec.point,
                "volume_min": spec.volume_min,
                "volume_max": spec.volume_max,
                "volume_step": spec.volume_step,
                "trade_mode": spec.trade_mode,
                "trade_mode_desc": _TRADE_MODE_DESC.get(spec.trade_mode, "UNKNOWN"),
                "trade_contract_size": spec.contract_size,
                "trade_tick_size": spec.tick_size,
                "trade_tick_value": spec.tick_value,
                "spec_source": spec.spec_source,
            })
        # ALWAYS emit bid/ask/age_seconds/stale so a caller can read info["bid"]
        # unconditionally. A known-but-closed market used to omit them entirely
        # (no tick within the window) -> info["bid"] KeyError'd consumers, while
        # get_symbol_price returned the frozen quote with a stale flag. Now the
        # keys are always present: a real or frozen tick -> its values + the same
        # staleness signal as get_symbol_price; no obtainable tick -> nulls
        # flagged stale, never dropped keys.
        if tick is not None:
            info["bid"] = tick.bid
            info["ask"] = tick.ask
            info["age_seconds"], info["stale"] = _tick_freshness(tick)
        else:
            info["bid"] = None
            info["ask"] = None
            info["age_seconds"] = None
            info["stale"] = True
        return info


class _Order:
    def __init__(self, owner: "TickerAllClient"):
        self._o = owner

    # — reads —
    def get_all_positions(self) -> pd.DataFrame:
        return _positions_to_df(self._o._positions())

    def get_positions_by_symbol(self, symbol: str) -> pd.DataFrame:
        return _positions_to_df([p for p in self._o._positions() if p.symbol == symbol])

    def get_positions_by_id(self, id: Union[int, str]) -> pd.DataFrame:
        return _positions_to_df([p for p in self._o._positions() if str(p.ticket) == str(id)])

    def get_all_pending_orders(self) -> pd.DataFrame:
        return _pending_to_df(self._o._pending())

    def get_pending_orders_by_symbol(self, symbol: str) -> pd.DataFrame:
        return _pending_to_df([o for o in self._o._pending() if o.symbol == symbol])

    def get_pending_orders_by_id(self, id: Union[int, str]) -> pd.DataFrame:
        return _pending_to_df([o for o in self._o._pending() if str(o.ticket) == str(id)])

    # — writes —
    def place_market_order(self, *, type: str, symbol: str, volume: Union[float, int]):
        # Market orders stay FAIL-FAST (no queue_if_reconnecting): on a cold
        # session the place fails cleanly rather than risk a fill at a stale
        # price; the caller re-decides with fresh prices. Do NOT add
        # queue_if_reconnecting here — it would queue-and-replay a price-
        # sensitive order.
        try:
            r = self._o._sdk.orders.place(
                self._o._account_id, type="market", symbol=symbol, side=type.upper(), volume=float(volume),
            )
            # The market result now carries the executed fill price directly
            # (the hosted API returns it on the fill), so no position re-read is
            # needed.
            return _ok("Order sent successfully", _order_result_data(r))
        except Exception as e:  # noqa: BLE001 — surface a clean message to the LLM
            return _err(str(e))

    def place_pending_order(self, *, type: str, symbol: str, volume: Union[float, int],
                            price: Union[float, int], stop_loss: Optional[Union[float, int]] = 0.0,
                            take_profit: Optional[Union[float, int]] = 0.0):
        try:
            kind = self._o._classify_pending(symbol, type.upper(), float(price))  # 'limit' | 'stop'
            # Pending orders are price-insensitive → queue-and-replay on a cold/
            # reconnecting session (the SDK reuses one idempotency key for the
            # replay). This makes the write exactly-once and prevents the
            # duplicate resting order a racy re-arm re-issue would create
            # (audit CRITICAL-1).
            r = self._o._sdk.orders.place(
                self._o._account_id, type=kind, symbol=symbol, side=type.upper(), volume=float(volume),
                price=float(price),
                stop_loss=(float(stop_loss) if stop_loss else None),
                take_profit=(float(take_profit) if take_profit else None),
                queue_if_reconnecting=True,
            )
            return _ok("Order sent successfully", _order_result_data(r))
        except Exception as e:  # noqa: BLE001
            return _err(str(e))

    def modify_position(self, id: Union[str, int], *, stop_loss: Optional[Union[int, float]] = None,
                        take_profit: Optional[Union[int, float]] = None):
        try:
            self._o._sdk.positions.modify(
                self._o._account_id, int(id),
                stop_loss=(float(stop_loss) if stop_loss is not None else None),
                take_profit=(float(take_profit) if take_profit is not None else None),
                queue_if_reconnecting=True,  # idempotent by ticket → safe to replay (audit HIGH-3)
            )
            return _ok(f"Modify position {id} success")
        except Exception as e:  # noqa: BLE001
            return _err(str(e))

    def modify_pending_order(self, *, id: Union[int, str], price: Optional[Union[int, float]] = None,
                             stop_loss: Optional[Union[int, float]] = None,
                             take_profit: Optional[Union[int, float]] = None):
        try:
            self._o._sdk.orders.modify_pending(
                self._o._account_id, int(id),
                price=(float(price) if price is not None else None),
                stop_loss=(float(stop_loss) if stop_loss is not None else None),
                take_profit=(float(take_profit) if take_profit is not None else None),
                queue_if_reconnecting=True,  # idempotent by ticket → safe to replay
            )
            return _ok(f"Modify pending order {id} success")
        except Exception as e:  # noqa: BLE001
            return _err(str(e))

    def close_position(self, id: Union[str, int]):
        try:
            # Idempotent by ticket → queue-and-replay on a cold session so a
            # dropped ack on a write that actually applied doesn't surface as a
            # false failure (audit HIGH-3 / TRADE_DROPPED).
            self._o._sdk.positions.close(self._o._account_id, int(id), queue_if_reconnecting=True)
            return _ok(f"Close position {id} success")
        except Exception as e:  # noqa: BLE001
            return _err(str(e))

    def cancel_pending_order(self, id: Union[int, str]):
        try:
            self._o._sdk.orders.cancel_pending(self._o._account_id, int(id), queue_if_reconnecting=True)
            return _ok(f"Cancel pending order {id} success")
        except Exception as e:  # noqa: BLE001
            return _err(str(e))

    # — bulk —
    def _close_positions(self, positions, label: str) -> Dict[str, Any]:
        n = 0
        for p in positions:
            try:
                self._o._sdk.positions.close(self._o._account_id, int(p.ticket), queue_if_reconnecting=True)
                n += 1
            except Exception:  # noqa: BLE001 — best-effort bulk close
                pass
        return _ok(f"Close {n} {label} success")

    def close_all_positions(self):
        return self._close_positions(self._o._positions(), "positions")

    def close_all_positions_by_symbol(self, symbol: str):
        return self._close_positions([p for p in self._o._positions() if p.symbol == symbol], f"{symbol} positions")

    def close_all_profitable_positions(self):
        return self._close_positions([p for p in self._o._positions() if (p.profit or 0) > 0], "profitable positions")

    def close_all_losing_positions(self):
        return self._close_positions([p for p in self._o._positions() if (p.profit or 0) < 0], "losing positions")

    def cancel_all_pending_orders(self):
        orders = self._o._pending()
        n = 0
        for o in orders:
            try:
                self._o._sdk.orders.cancel_pending(self._o._account_id, int(o.ticket), queue_if_reconnecting=True)
                n += 1
            except Exception:  # noqa: BLE001
                pass
        return _ok(f"Cancel {n} pending orders success")

    def cancel_pending_orders_by_symbol(self, symbol: str):
        orders = [o for o in self._o._pending() if o.symbol == symbol]
        n = 0
        for o in orders:
            try:
                self._o._sdk.orders.cancel_pending(self._o._account_id, int(o.ticket), queue_if_reconnecting=True)
                n += 1
            except Exception:  # noqa: BLE001
                pass
        return _ok(f"Cancel {n} {symbol} pending orders success")


class _History:
    def __init__(self, owner: "TickerAllClient"):
        self._o = owner

    def get_deals_as_dataframe(self, from_date: Optional[str] = None, to_date: Optional[str] = None,
                               group: Optional[str] = None) -> pd.DataFrame:
        trades = self._o._sdk.history.get(self._o._account_id, symbol=group,
                                          from_=from_date, to=_inclusive_to(to_date))
        return _history_to_df(trades)

    def get_orders_as_dataframe(self, from_date: Optional[str] = None, to_date: Optional[str] = None,
                                group: Optional[str] = None) -> pd.DataFrame:
        """Filled-ORDER history (one row per filled order) — a real order log,
        distinct from the round-trip DEAL view of get_deals, so a bot can
        reconcile intent vs fills (audit §8/§3). Filled-only: cancelled/rejected/
        expired orders aren't returned by the API. Falls back to the deal view on
        an older SDK that doesn't expose the orders endpoint."""
        orders_fn = getattr(getattr(self._o._sdk, "history", None), "orders", None)
        if callable(orders_fn):
            orders = orders_fn(self._o._account_id, symbol=group,
                               from_=from_date, to=_inclusive_to(to_date))
            return _orders_to_df(orders)
        return self.get_deals_as_dataframe(from_date=from_date, to_date=to_date, group=group)


def _order_result_data(r) -> Dict[str, Any]:
    return {
        "ticket": r.ticket, "symbol": r.symbol, "side": r.side, "type": r.type,
        "volume": r.volume, "status": r.status, "price": r.price,
        "stop_loss": r.stop_loss, "take_profit": r.take_profit,
    }


# ── Main client ───────────────────────────────────────────────────────────────

class TickerAllClient:
    """Drop-in alternative to ``MT5Client`` backed by a hosted MetaTrader API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self._api_key = cfg.get("api_key") or os.getenv("TICKERALL_API_KEY")
        self._base_url = cfg.get("base_url") or os.getenv("TICKERALL_API_BASE_URL")
        # Explicit WS stream URL, or derived from base_url at connect() so live
        # ticks follow a custom (self-hosted / staging) host instead of prod.
        self._stream_url = cfg.get("stream_url") or os.getenv("TICKERALL_STREAM_URL")
        self._broker = cfg.get("broker") or os.getenv("TICKERALL_BROKER", "mt5")
        self._server = cfg.get("server")
        self._account = cfg.get("account") or cfg.get("login")
        self._password = cfg.get("password")

        self._sdk = None
        self._account_id: Optional[str] = None
        self._connected = False

        # Live tick stream. The SDK maintains the latest-tick cache and the
        # per-symbol subscription state itself (wait_for_tick), so the adapter
        # no longer keeps its own tick dict / subscribed set.
        self._stream = None
        self._spec_cache: Optional[Dict[str, Any]] = None
        self._spec_cache_at: float = 0.0
        self._symbols_cache: Optional[set] = None
        self._symbols_cache_at: float = 0.0
        # Re-fetch the symbol/spec maps after this TTL (and on every reconnect)
        # so a refreshed catalog — a hosted-service redeploy or a broker spec
        # change — is picked up without restarting the process. Previously these were
        # cached once-and-forever, so a session that outlived a catalog change
        # served stale specs (e.g. a derived volume_min) until a manual restart.
        self._spec_cache_ttl: float = 300.0
        self._lock = threading.Lock()

        self.account = _Account(self)
        self.market = _Market(self)
        self.order = _Order(self)
        self.history = _History(self)

    # — connection —
    def connect(self) -> bool:
        if not self._api_key:
            raise ConnectionError("TICKERALL_API_KEY is required for the hosted provider.")
        try:
            from tickerall import Tickerall
        except ImportError as e:  # pragma: no cover
            raise ConnectionError(
                "The 'tickerall' package is required for the hosted provider. Install it with: pip install tickerall"
            ) from e

        kwargs: Dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        # Only override the WS stream URL when one is explicitly set (env).
        # Otherwise the SDK derives it from base_url (http→ws, https→wss), so a
        # custom / staging host streams ticks from itself, not the prod default.
        if self._stream_url:
            kwargs["stream_url"] = self._stream_url
        self._sdk = Tickerall(**kwargs)

        if not (self._server and self._account and self._password):
            raise ConnectionError("Hosted provider needs server, account (login) and password to open a session.")

        # keep_alive caches the credentials in this process (never persisted) so
        # the connection transparently re-arms if it later goes cold — e.g. across
        # a broker drop or a hosted-side restart. The next tool call after a cold
        # auto-reconnects, so a long-running MCP/HTTP server self-heals without a
        # manual reconnect. (A local MT5 terminal has no equivalent.)
        result = self._sdk.sessions.keep_alive(
            broker=self._broker, server=self._server,
            account=int(self._account), password=self._password,
        )
        self._account_id = result.account_id
        self._connected = True
        # A (re)connect may bring a refreshed catalog (hosted-service redeploy /
        # broker spec change), so drop the cached symbol/spec maps — they
        # re-populate lazily from the fresh session.
        self._spec_cache = None
        self._symbols_cache = None
        return True

    def disconnect(self) -> bool:
        self._connected = False
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        # End the hosted session (also drops the kept credentials). Best-effort —
        # if it fails, the session simply cools on its own TTL.
        if self._sdk is not None and self._account_id is not None:
            try:
                self._sdk.sessions.end(self._account_id)
            except Exception:  # noqa: BLE001
                pass
        return True

    def is_connected(self) -> bool:
        return self._connected

    def get_terminal_info(self) -> Dict[str, Any]:
        return {"provider": "tickerall", "server": self._server, "account_id": self._account_id}

    def get_version(self):
        return (5, 0, 0, 0)

    def last_error(self):
        return (0, "no error")

    # — internals —
    def _require_online(self):
        """Fetch the account snapshot and FAIL LOUD if the broker connection is
        still warming/offline. Without this guard, position / pending reads
        return an empty list during the ~90s cold-start window — which is
        indistinguishable from a genuinely flat account. A bot polling at
        startup or right after a reconnect would then see 'no positions' when
        the truth is UNKNOWN, and could open a duplicate or skip stop-loss
        management (audit BLOCKER §3). `get_account_info` already errors in this
        state via `info is None`; the read paths must be consistent."""
        snap = self._sdk.accounts.get(self._account_id)
        if snap.account is None or getattr(snap, "status", "online") != "online":
            raise RuntimeError(
                "Account is warming/offline — broker position state is UNKNOWN "
                "(not flat). Retry shortly; do not treat an empty read as "
                "'no positions'."
            )
        return snap

    def _positions(self):
        return self._require_online().positions

    def _pending(self):
        self._require_online()
        return self._sdk.orders.list_pending(self._account_id)

    def _symbol_known(self, name: str) -> bool:
        """True if the symbol is offered on this account. Lets callers tell an
        UNKNOWN symbol (typo / unsupported) apart from a known-but-closed
        market (audit §7) — the two must not return the same error, or a bot
        retries forever on a symbol that will never exist."""
        now = time.monotonic()
        if self._symbols_cache is None or (now - self._symbols_cache_at) > self._spec_cache_ttl:
            try:
                self._symbols_cache = set(self._sdk.accounts.symbols(self._account_id))
                self._symbols_cache_at = now
            except Exception:  # noqa: BLE001
                # Keep any prior set on a transient refresh failure (only fall
                # back to empty on the first fetch); back off a TTL before retry.
                if self._symbols_cache is None:
                    self._symbols_cache = set()
                self._symbols_cache_at = now
        return name in self._symbols_cache

    def _symbol_spec(self, name: str):
        """Look up one symbol's spec, caching the account's full spec list on
        first use (it's effectively static for a session)."""
        now = time.monotonic()
        if self._spec_cache is None or (now - self._spec_cache_at) > self._spec_cache_ttl:
            try:
                specs = self._sdk.accounts.symbol_specs(self._account_id)
                self._spec_cache = {s.name: s for s in specs}
                self._spec_cache_at = now
            except Exception:  # noqa: BLE001
                # Keep the prior spec map on a transient refresh failure (only
                # fall back to empty on the first fetch); back off a TTL.
                if self._spec_cache is None:
                    self._spec_cache = {}
                self._spec_cache_at = now
        return self._spec_cache.get(name)

    def _ensure_stream(self):
        if self._stream is not None:
            return
        with self._lock:
            if self._stream is not None:
                return
            # The SDK stream maintains the latest-tick cache itself — no
            # on("tick") wiring needed here.
            self._stream = self._sdk.stream.connect()

    def _latest_tick(self, symbol: str, timeout: float = 4.0):
        self._ensure_stream()
        # wait_for_tick subscribes the symbol if needed (a no-op once subscribed,
        # and it re-subscribes correctly on a fresh stream after a reconnect),
        # caches inbound ticks, and blocks up to `timeout` for the first one.
        try:
            return self._stream.wait_for_tick(symbol, account_id=self._account_id, timeout=timeout)
        except TimeoutError as e:
            # Preserve the RuntimeError contract get_symbol_price / get_symbol_info
            # rely on (a closed market with no tick reads as "no price").
            raise RuntimeError(
                f"No price received for '{symbol}' (no recent tick — market may be closed)."
            ) from e

    def _classify_pending(self, symbol: str, side: str, price: float) -> str:
        """Map a (side, price) pending order to a 'limit' or 'stop' relative to
        the current market — the broker semantics MT5 infers automatically."""
        try:
            tick = self._latest_tick(symbol, timeout=4.0)
        except Exception:
            return "limit"
        if side == "BUY":
            return "limit" if price < tick.ask else "stop"
        return "limit" if price > tick.bid else "stop"
