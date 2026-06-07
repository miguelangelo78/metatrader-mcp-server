"""
Offline tests for the optional hosted (TickerAll) provider.

These inject a fake SDK + fake response objects (SimpleNamespace), so they need
only pandas — no network, no live account, and no `tickerall` install. They lock
in the MT5-compatible shapes the MCP tools consume (DataFrame columns, the
{error, message, data} order-result envelope, the symbol filter, bulk closes).
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from metatrader_client.tickerall_client import TickerAllClient


def _position(**kw):
    base = dict(ticket=1, symbol="BTCUSDm", side="BUY", volume=0.10, stop_loss=0.0,
                take_profit=0.0, magic=0, comment="", swap=0.0, commission=0.0,
                open_time="2026-06-05T10:00:00Z", entry_price=62000.0, current_price=62100.0,
                profit=10.0, last_update=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _candle(ts, o, h, l, c, tick_volume=5, spread=0.5):
    return SimpleNamespace(timestamp=ts, open=o, high=h, low=l, close=c, bid=c,
                           tick_volume=tick_volume, spread=spread)


def _horder(**kw):
    base = dict(order_ticket="9001", symbol="BTCUSDm", side="BUY", volume=0.1,
                price=62000.0, time="2026-06-05T10:00:00Z", position_id="9001",
                state="FILLED", deal_count=1)
    base.update(kw)
    return SimpleNamespace(**base)


class _Recorder:
    """Captures the last call's kwargs so tests can assert the mapping."""
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(ticket=999, symbol=kwargs.get("symbol", ""), side=kwargs.get("side", ""),
                               type=kwargs.get("type", ""), volume=kwargs.get("volume", 0.0),
                               status="open", price=kwargs.get("price"), stop_loss=None, take_profit=None)


def _spec(**kw):
    base = dict(name="BTCUSDm", volume_min=0.01, volume_max=100.0, volume_step=0.01,
                spec_source="broker", trade_mode=4, digits=2, point=0.01, tick_size=0.01,
                contract_size=1.0, tick_value=0.01)
    base.update(kw)
    return SimpleNamespace(**base)


class _FakeStream:
    """Stand-in for the SDK's TickerallStream. `_ensure_stream` short-circuits
    when `c._stream` is set, and `_latest_tick` delegates to `wait_for_tick`, so
    a test just seeds the per-symbol ticks here (no real WebSocket)."""
    def __init__(self, ticks=None):
        self._ticks = dict(ticks or {})

    def wait_for_tick(self, symbol, *, account_id=None, timeout=4.0):
        tick = self._ticks.get(symbol)
        if tick is None:
            raise TimeoutError(f"no tick for {symbol}")
        return tick

    def latest_tick(self, symbol):
        return self._ticks.get(symbol)

    def close(self):
        pass


def _client_with_sdk(positions=None, pending=None, candles=None, symbols=None, specs=None, warming=False):
    c = TickerAllClient({"api_key": "x", "server": "S", "account": 1, "password": "p"})
    c._account_id = "acc1"
    c._connected = True
    place = _Recorder()
    closed, cancelled = [], []
    # Default symbol universe so `_symbol_known` (the §7 unknown-vs-closed guard)
    # passes for the common symbols the tests price; pass `symbols=` to override.
    sym_list = symbols if symbols is not None else ["BTCUSDm", "ETHUSDm", "EURUSDm", "XAUUSDm"]

    def _snapshot(account_id, **k):
        if warming:
            # Cold-start window: broker state not yet known → account is None,
            # status offline. Reads must error, not report flat (audit §3).
            return SimpleNamespace(positions=[], account=None, status="offline")
        return SimpleNamespace(
            positions=list(positions or []), status="online",
            account=SimpleNamespace(name="Demo", account_type="demo", currency="USD", leverage=500,
                                    balance=1000.0, equity=1010.0, margin=5.0, free_margin=1005.0,
                                    margin_level=200.0),
        )

    sdk = SimpleNamespace(
        accounts=SimpleNamespace(
            symbols=lambda account_id, **k: list(sym_list),
            symbol_specs=lambda account_id, **k: list(specs or []),
            get=_snapshot,
        ),
        orders=SimpleNamespace(
            list_pending=lambda account_id, **k: list(pending or []),
            place=place,
            cancel_pending=lambda account_id, ticket, **k: cancelled.append(ticket),
        ),
        positions=SimpleNamespace(
            close=lambda account_id, ticket, **k: closed.append(ticket),
            modify=lambda account_id, ticket, **k: None,
        ),
        candles=SimpleNamespace(get=lambda account_id, **k: list(candles or [])),
        history=SimpleNamespace(get=lambda account_id, **k: [], orders=lambda account_id, **k: []),
    )
    c._sdk = sdk
    return c, place, closed, cancelled


def test_positions_dataframe_has_mt5_columns():
    c, *_ = _client_with_sdk(positions=[_position(ticket=11), _position(ticket=22, side="SELL")])
    df = c.order.get_all_positions()
    assert len(df) == 2
    for col in ("ticket", "type", "volume", "price_open", "sl", "tp", "price_current", "profit", "symbol"):
        assert col in df.columns
    assert set(df["type"]) == {"BUY", "SELL"}


def test_get_positions_by_symbol_and_id_filter():
    c, *_ = _client_with_sdk(positions=[_position(ticket=11, symbol="BTCUSDm"),
                                        _position(ticket=22, symbol="EURUSDm")])
    assert list(c.order.get_positions_by_symbol("EURUSDm")["ticket"]) == [22]
    assert list(c.order.get_positions_by_id(11)["ticket"]) == [11]
    assert c.order.get_positions_by_symbol("XAUUSDm").empty


def test_place_market_order_maps_to_sdk_and_wraps_result():
    c, place, *_ = _client_with_sdk()
    res = c.order.place_market_order(type="buy", symbol="BTCUSDm", volume=0.5)
    assert res["error"] is False and res["data"]["ticket"] == 999
    _, kwargs = place.calls[-1]
    assert kwargs["type"] == "market" and kwargs["side"] == "BUY" and kwargs["volume"] == 0.5


def test_place_market_order_error_is_caught():
    c, *_ = _client_with_sdk()
    def boom(*a, **k):
        raise RuntimeError("broker rejected")
    c._sdk.orders.place = boom
    res = c.order.place_market_order(type="BUY", symbol="BTCUSDm", volume=0.1)
    assert res["error"] is True and "broker rejected" in res["message"]


def test_close_all_positions_iterates_every_ticket():
    c, _place, closed, _ = _client_with_sdk(positions=[_position(ticket=11), _position(ticket=22), _position(ticket=33)])
    res = c.order.close_all_positions()
    assert res["error"] is False and sorted(closed) == [11, 22, 33]


def test_close_all_profitable_only_closes_winners():
    c, _place, closed, _ = _client_with_sdk(positions=[_position(ticket=11, profit=5.0),
                                                       _position(ticket=22, profit=-3.0)])
    c.order.close_all_profitable_positions()
    assert closed == [11]


def test_get_symbols_group_filter():
    c, *_ = _client_with_sdk(symbols=["BTCUSDm", "EURUSDm", "EURGBPm", "EURJPYm"])
    assert c.market.get_symbols() == ["BTCUSDm", "EURUSDm", "EURGBPm", "EURJPYm"]
    # *USD* keeps the two USD pairs; EUR/GBP and EUR/JPY are excluded.
    assert c.market.get_symbols(group="*USD*") == ["BTCUSDm", "EURUSDm"]
    assert c.market.get_symbols(group="EUR*") == ["EURUSDm", "EURGBPm", "EURJPYm"]


def test_candles_dataframe_shape_and_sort():
    c, *_ = _client_with_sdk(candles=[_candle(1000, 1, 2, 0.5, 1.5, tick_volume=7, spread=0.3),
                                      _candle(4600, 1.5, 2.5, 1, 2.0, tick_volume=12, spread=0.4)])
    df = c.market.get_candles_latest("BTCUSDm", "H1", 10)
    assert list(df.columns) == ["time", "open", "high", "low", "close", "tick_volume", "spread"]
    # newest first
    assert df.iloc[0]["close"] == 2.0
    # real tick_volume + spread surfaced (audit §4: were hardcoded 0 / absent)
    assert df.iloc[0]["tick_volume"] == 12 and df.iloc[0]["spread"] == pytest.approx(0.4)
    assert str(df["time"].dtype).startswith("datetime64")


def test_get_orders_uses_real_order_log_distinct_from_deals():
    # Audit §8: get_orders must hit the real FILLED-order log, not alias deals.
    c, *_ = _client_with_sdk()
    c._sdk.history = SimpleNamespace(
        get=lambda account_id, **k: [_trade()],   # the deal view
        orders=lambda account_id, **k: [_horder(order_ticket="9001"), _horder(order_ticket="9002")],
    )
    df = c.history.get_orders_as_dataframe()
    assert list(df["order_ticket"]) == ["9001", "9002"]
    assert "order_ticket" in df.columns and "position_id" in df.columns


def test_get_orders_falls_back_to_deals_on_older_sdk():
    # Graceful fallback: an SDK without history.orders → the deal view (no crash).
    c, *_ = _client_with_sdk()
    c._sdk.history = SimpleNamespace(get=lambda account_id, **k: [_trade(ticket="111")])
    df = c.history.get_orders_as_dataframe()
    assert list(df["ticket"]) == ["111"]


def test_place_market_order_surfaces_fill_price_from_result():
    # Audit §4 closed: the market result now carries the executed fill price
    # directly (the hosted API returns it on the fill), so the adapter passes it
    # through and does NOT re-read the position.
    c, *_ = _client_with_sdk(positions=[_position(ticket=999, entry_price=62050.0)])
    # Simulate the hosted API returning the fill price on the market result.
    c._sdk.orders.place = lambda account_id, **k: SimpleNamespace(
        ticket=999, symbol=k.get("symbol", ""), side=k.get("side", ""), type=k.get("type", ""),
        volume=k.get("volume", 0.0), status="open", price=1632.5, stop_loss=None, take_profit=None)
    res = c.order.place_market_order(type="BUY", symbol="ETHUSDm", volume=0.1)
    # Returns the RESULT's fill price (1632.5), not the position's entry (62050).
    assert res["error"] is False and res["data"]["price"] == 1632.5


def test_account_trade_statistics_shape():
    c, *_ = _client_with_sdk()
    info = c.account.get_trade_statistics()
    for k in ("balance", "equity", "profit", "margin_level", "free_margin", "account_type", "leverage", "currency"):
        assert k in info
    assert info["account_type"] == "demo"
    assert info["profit"] == pytest.approx(10.0)  # equity 1010 - balance 1000


def test_get_symbol_price_reads_tick_cache():
    c, *_ = _client_with_sdk()
    # Inject a fake stream whose wait_for_tick returns a cached tick (no real WS).
    c._stream = _FakeStream({"BTCUSDm": SimpleNamespace(bid=62000.0, ask=62010.0, timestamp="2026-06-05T10:00:00Z")})
    price = c.market.get_symbol_price("BTCUSDm")
    assert price["bid"] == 62000.0 and price["ask"] == 62010.0 and price["last"] == 62000.0


def test_get_symbol_info_maps_spec_and_tick():
    c, *_ = _client_with_sdk(specs=[_spec(name="BTCUSDm", volume_min=0.01, volume_step=0.01, digits=2)])
    c._stream = _FakeStream({"BTCUSDm": SimpleNamespace(bid=62000.0, ask=62010.0, timestamp="2026-06-05T10:00:00Z")})
    info = c.market.get_symbol_info("BTCUSDm")
    assert info["name"] == "BTCUSDm"
    for k in ("digits", "point", "volume_min", "volume_max", "volume_step", "trade_mode",
              "trade_contract_size", "trade_tick_size", "trade_tick_value", "bid", "ask"):
        assert k in info
    assert info["bid"] == 62000.0 and info["volume_step"] == 0.01


def test_get_symbol_info_embeds_quote_freshness():
    # Re-audit nit: the bid/ask embedded in get_symbol_info must carry the same
    # stale flag as get_symbol_price, so a caller can't mistake a frozen weekend
    # quote for a live one.
    c, *_ = _client_with_sdk(specs=[_spec(name="EURUSDm", digits=5)], symbols=["EURUSDm"])
    c._stream = _FakeStream({"EURUSDm": SimpleNamespace(bid=1.152, ask=1.164, timestamp="2026-06-04T20:57:00Z")})
    info = c.market.get_symbol_info("EURUSDm")
    assert info["stale"] is True and info["age_seconds"] > 60


def test_get_symbol_info_includes_trade_mode_label():
    # Polish: surface a human label next to the raw ENUM_SYMBOL_TRADE_MODE int.
    c, *_ = _client_with_sdk(specs=[_spec(name="BTCUSDm", trade_mode=4)], symbols=["BTCUSDm"])
    c._stream = _FakeStream({"BTCUSDm": SimpleNamespace(bid=1.0, ask=1.1, timestamp="2026-06-05T10:00:00Z")})
    info = c.market.get_symbol_info("BTCUSDm")
    assert info["trade_mode"] == 4 and info["trade_mode_desc"] == "FULL"


def test_get_symbol_info_unknown_symbol_raises():
    # No spec and the fake SDK has no stream, so the tick fetch fails fast → raise.
    c, *_ = _client_with_sdk(specs=[])
    import pytest as _pt
    with _pt.raises(Exception):
        c.market.get_symbol_info("FOOBARm")


def test_get_symbol_info_always_emits_price_keys_even_with_no_tick():
    # A known-but-closed market with no obtainable tick used to DROP bid/ask/
    # stale entirely, so a caller reading info["bid"] hit a KeyError (while
    # get_symbol_price returned a frozen quote + stale flag). The keys must now
    # always be present: nulls flagged stale rather than omitted.
    c, *_ = _client_with_sdk(specs=[_spec(name="XAUUSDm", digits=3)], symbols=["XAUUSDm"])
    def _no_tick(symbol, timeout=4.0):
        raise RuntimeError("No price received (market may be closed).")
    c._latest_tick = _no_tick  # _Market reads self._o._latest_tick
    info = c.market.get_symbol_info("XAUUSDm")
    assert info["digits"] == 3 and info["trade_mode_desc"] == "FULL"  # spec still served
    for k in ("bid", "ask", "age_seconds", "stale"):
        assert k in info  # never dropped
    assert info["bid"] is None and info["ask"] is None and info["stale"] is True


def test_get_candles_latest_unknown_symbol_raises_not_empty():
    # Audit §7 parity: a typo'd / unsupported symbol must ERROR (like
    # get_symbol_price), not silently return an empty frame.
    c, *_ = _client_with_sdk(symbols=["BTCUSDm", "ETHUSDm"])
    with pytest.raises(RuntimeError, match="unknown symbol"):
        c.market.get_candles_latest("NOTAREALSYMBOLm", "H1", 10)
    # a known symbol with no bars in the window stays a (legitimate) empty frame
    assert c.market.get_candles_latest("BTCUSDm", "H1", 10).empty


def test_get_candles_by_date_unknown_symbol_raises():
    c, *_ = _client_with_sdk(symbols=["BTCUSDm"])
    with pytest.raises(RuntimeError, match="unknown symbol"):
        c.market.get_candles_by_date("NOPEm", "H1", from_date="2026-06-01")


def test_spec_cache_refreshes_not_cached_once_forever():
    # ETHUSDm-derived-spec staleness fix: the spec cache must refresh on
    # reconnect (connect() clears it) and after a TTL, so a refreshed catalog
    # (hosted-service redeploy / broker spec change) is picked up without a
    # restart — not cached once-and-forever like before.
    import time as _t
    c, *_ = _client_with_sdk(specs=[_spec(name="ETHUSDm", volume_min=0.01)])
    assert c._symbol_spec("ETHUSDm").volume_min == 0.01            # first fetch (stale/derived min)
    # broker refresh: the SDK now returns the corrected spec
    c._sdk.accounts.symbol_specs = lambda account_id, **k: [_spec(name="ETHUSDm", volume_min=0.1)]
    assert c._symbol_spec("ETHUSDm").volume_min == 0.01            # within TTL, no reconnect -> still cached
    c._spec_cache = None                                           # reconnect clears it (connect() does this)
    assert c._symbol_spec("ETHUSDm").volume_min == 0.1            # -> re-fetches the corrected min
    # TTL path: re-stale + expire the timestamp -> re-fetch
    c._spec_cache = {"ETHUSDm": _spec(name="ETHUSDm", volume_min=0.01)}
    c._spec_cache_at = _t.monotonic() - c._spec_cache_ttl - 1
    assert c._symbol_spec("ETHUSDm").volume_min == 0.1


def test_spec_cache_keeps_prior_on_transient_refresh_failure():
    # A blip during a TTL refresh must NOT blank a working cache.
    import time as _t
    c, *_ = _client_with_sdk(specs=[_spec(name="ETHUSDm", volume_min=0.1)])
    assert c._symbol_spec("ETHUSDm").volume_min == 0.1
    def _boom(*a, **k): raise RuntimeError("hosted API blip")
    c._sdk.accounts.symbol_specs = _boom
    c._spec_cache_at = _t.monotonic() - c._spec_cache_ttl - 1      # force a refresh attempt
    assert c._symbol_spec("ETHUSDm").volume_min == 0.1             # prior spec preserved, not blanked


def test_disconnect_ends_session_and_closes_stream():
    c = TickerAllClient({"api_key": "x", "server": "S", "account": 1, "password": "p"})
    c._account_id = "acc1"; c._connected = True
    ended, closed = [], []
    c._sdk = SimpleNamespace(sessions=SimpleNamespace(end=lambda aid, **k: ended.append(aid)))
    c._stream = SimpleNamespace(close=lambda: closed.append(True))
    assert c.disconnect() is True
    assert ended == ["acc1"] and closed == [True]
    assert c._connected is False and c._stream is None


def test_account_full_surface():
    c, *_ = _client_with_sdk()  # fake account: balance 1000, equity 1010, margin 5, free 1005, ml 200, lev 500
    assert c.account.get_balance() == 1000.0
    assert c.account.get_equity() == 1010.0
    assert c.account.get_margin() == 5.0
    assert c.account.get_free_margin() == 1005.0
    assert c.account.get_margin_level() == 200.0
    assert c.account.get_currency() == "USD"
    assert c.account.get_leverage() == 500
    assert c.account.get_account_type() == "demo"
    assert c.account.is_trade_allowed() is True
    assert c.account.check_margin_level(100.0) is True


def _trade(**kw):
    base = dict(ticket="111", symbol="ETHUSDm", side="SELL", volume=0.01, open_price=60748.35,
                close_price=60864.65, open_time="t1", close_time="t2", profit=-1.163, swap=0.0, commission=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_queue_if_reconnecting_on_idempotent_ops_but_not_market():
    # The at-least-once safety fix: price-insensitive / idempotent writes
    # queue-and-replay on a cold session; market orders stay fail-fast.
    c = TickerAllClient({"api_key": "x", "server": "S", "account": 1, "password": "p"})
    c._account_id = "acc1"; c._connected = True
    captured = {}
    def rec(name):
        def f(*a, **k):
            captured[name] = k
            return SimpleNamespace(ticket=1, symbol="BTCUSDm", side="BUY", type="market",
                                   volume=0.1, status="open", price=None, stop_loss=None, take_profit=None)
        return f
    # _classify_pending needs a tick — inject a fake stream that returns one.
    c._stream = _FakeStream({"BTCUSDm": SimpleNamespace(bid=100.0, ask=101.0, timestamp="t")})
    c._sdk = SimpleNamespace(
        orders=SimpleNamespace(place=rec("place"), cancel_pending=rec("cancel"), modify_pending=rec("modpend")),
        positions=SimpleNamespace(close=rec("close"), modify=rec("modpos")),
    )
    c.order.place_market_order(type="BUY", symbol="BTCUSDm", volume=0.1)
    assert captured["place"].get("queue_if_reconnecting") in (None, False)  # market = fail-fast
    c.order.place_pending_order(type="BUY", symbol="BTCUSDm", volume=0.1, price=99.0)  # below ask → limit
    assert captured["place"].get("queue_if_reconnecting") is True           # pending = queued
    c.order.close_position(5);          assert captured["close"].get("queue_if_reconnecting") is True
    c.order.cancel_pending_order(5);    assert captured["cancel"].get("queue_if_reconnecting") is True
    c.order.modify_position(5, stop_loss=90.0);   assert captured["modpos"].get("queue_if_reconnecting") is True
    c.order.modify_pending_order(id=5, price=98.0); assert captured["modpend"].get("queue_if_reconnecting") is True


def test_history_drops_only_identical_duplicate_rows():
    from metatrader_client.tickerall_client import _history_to_df
    a = _trade()
    # the same round-trip surfaced twice (entry + exit deals project identically) -> one row
    assert len(_history_to_df([a, a])) == 1
    # a genuinely distinct trade is kept
    b = _trade(ticket="222", symbol="BTCUSDm", side="BUY", volume=0.05, profit=5.0)
    df = _history_to_df([a, a, b])
    assert len(df) == 2
    assert set(df["ticket"]) == {"111", "222"}


def test_to_date_is_inclusive_of_the_whole_day():
    # Audit §1: a bare YYYY-MM-DD `to` (esp. a same-day from==to query) must
    # cover the whole day, not filter to midnight and silently return empty.
    from metatrader_client.tickerall_client import _inclusive_to
    assert _inclusive_to("2026-06-06") == "2026-06-06T23:59:59.999999"
    assert _inclusive_to("2026-06-06T12:30:00") == "2026-06-06T12:30:00"  # time given → untouched
    assert _inclusive_to(None) is None
    # and the deals path passes the bumped `to` through to the SDK
    captured = {}
    c, *_ = _client_with_sdk()
    c._sdk.history = SimpleNamespace(get=lambda account_id, **k: captured.update(k) or [])
    c.history.get_deals_as_dataframe(from_date="2026-06-06", to_date="2026-06-06")
    assert captured["to"] == "2026-06-06T23:59:59.999999" and captured["from_"] == "2026-06-06"


def test_stream_url_passthrough_lets_sdk_derive():
    # The provider no longer derives the WS URL itself — the SDK derives it from
    # base_url (http→ws / https→wss + /v1/stream; covered by the SDK's own tests).
    # The provider only forwards an explicit TICKERALL_STREAM_URL override; with
    # none set, _stream_url is None so connect() doesn't pass stream_url and the
    # SDK derives it from base_url.
    import os
    c = TickerAllClient({"api_key": "x", "base_url": "http://localhost:3100"})
    assert c._stream_url is None
    # explicit TICKERALL_STREAM_URL wins (forwarded verbatim to the SDK).
    os.environ["TICKERALL_STREAM_URL"] = "wss://custom/v1/stream"
    try:
        c2 = TickerAllClient({"api_key": "x", "base_url": "http://localhost:3100"})
        assert c2._stream_url == "wss://custom/v1/stream"
    finally:
        del os.environ["TICKERALL_STREAM_URL"]


def test_reads_raise_during_warming_instead_of_reporting_flat():
    # Audit BLOCKER §3: during the cold-start window, position/pending reads must
    # ERROR (state unknown), not silently return an empty "flat" frame. Otherwise
    # a bot polling at startup opens a duplicate or skips stop-loss management.
    c, *_ = _client_with_sdk(warming=True)
    with pytest.raises(RuntimeError, match="warming"):
        c.order.get_all_positions()
    with pytest.raises(RuntimeError, match="warming"):
        c.order.get_all_pending_orders()
    # The account read already errors gracefully (returns an envelope, not raise).
    info = c.account.get_trade_statistics()
    assert info.get("error") is True
    # And once online, an empty position list is a legitimate "flat" (no raise).
    c2, *_ = _client_with_sdk(positions=[])
    assert c2.order.get_all_positions().empty


def test_get_symbol_price_unknown_symbol_is_distinct_from_closed_market():
    # Audit §7: a typo / unsupported symbol must not return the same "no recent
    # tick — market may be closed" error as a real-but-closed symbol.
    c, *_ = _client_with_sdk(symbols=["BTCUSDm", "ETHUSDm"])
    with pytest.raises(RuntimeError, match="unknown symbol"):
        c.market.get_symbol_price("NOTAREALSYMBOLm")


def test_get_symbol_price_flags_stale_weekend_quote():
    # Audit §14: a frozen Friday-close quote must be flagged stale so a 24/5 bot
    # can refuse it. A fresh quote must not be flagged.
    from datetime import datetime, timezone
    c, *_ = _client_with_sdk(symbols=["ETHUSDm"])
    stream = _FakeStream()
    c._stream = stream
    # Stale: a 2-day-old tick.
    stream._ticks["ETHUSDm"] = SimpleNamespace(bid=1561.0, ask=1700.0, timestamp="2026-06-04T20:57:00Z")
    stale = c.market.get_symbol_price("ETHUSDm")
    assert stale["stale"] is True and stale["age_seconds"] is not None and stale["age_seconds"] > 60
    # Fresh: a tick stamped ~now.
    stream._ticks["ETHUSDm"] = SimpleNamespace(bid=1561.0, ask=1561.5,
                                               timestamp=datetime.now(timezone.utc).isoformat())
    fresh = c.market.get_symbol_price("ETHUSDm")
    assert fresh["stale"] is False


def test_history_coerces_blank_profit_to_zero():
    # Audit §2: an exit-deal leg can carry a blank profit/swap/commission; coerce
    # to 0.0 so a CSV consumer parsing float doesn't hit NaN / an exception.
    from metatrader_client.tickerall_client import _history_to_df
    leg = _trade(ticket="333", profit=None, swap=None, commission=None)
    df = _history_to_df([leg])
    assert df.iloc[0]["profit"] == 0.0
    assert df.iloc[0]["swap"] == 0.0
    assert df.iloc[0]["commission"] == 0.0
