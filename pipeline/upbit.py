"""Utilities for downloading market data from the Upbit quotation APIs.

The module wraps the public REST endpoints that do not require
authentication and returns the responses as :class:`pandas.DataFrame`
objects that integrate nicely with the rest of the research pipeline.

Only the quotation (시세) endpoints are implemented here because they are
enough for constructing pricing datasets.  Authentication for account or
trading endpoints would require managing API keys and is therefore kept
out of scope for this helper module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Literal, Optional, Sequence

import pandas as pd
import requests


Interval = Literal["seconds", "minutes", "days", "weeks", "months", "years"]


def _format_to_param(value: Optional[datetime | str]) -> Optional[str]:
    """Convert ``to`` parameter into the format expected by Upbit.

    The REST documentation specifies a ``YYYY-MM-DD HH:MM:SS`` timestamp
    string.  When the user provides a :class:`datetime.datetime` we convert
    it into UTC and format it accordingly.  ``None`` values are passed
    through unchanged because the parameter is optional.
    """

    if value is None or isinstance(value, str):
        return value

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class UpbitClient:
    """Small helper around the public Upbit quotation REST API.

    Parameters
    ----------
    base_url:
        Base endpoint for the API.  The default points at the Korean
        production cluster (``https://api.upbit.com``) but the client also
        works with regional deployments such as the Singapore one by simply
        swapping the domain.
    session:
        Optional :class:`requests.Session` instance for connection reuse.
    timeout:
        Timeout in seconds for HTTP requests.
    """

    base_url: str = "https://api.upbit.com"
    session: Optional[requests.Session] = None
    timeout: float = 10.0

    def _request(self, path: str, params: Optional[dict] = None) -> list | dict:
        url = f"{self.base_url.rstrip('/')}{path}"
        sess = self.session or requests.Session()
        response = sess.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Public quotation endpoints
    # ------------------------------------------------------------------
    def list_markets(self, is_details: bool = False) -> pd.DataFrame:
        """Return the universe of supported trading pairs.

        Parameters
        ----------
        is_details:
            When ``True`` the API returns additional metadata such as the
            trading state and delisting schedule.
        """

        payload = self._request(
            "/v1/market/all",
            params={"isDetails": str(is_details).lower()},
        )
        return pd.DataFrame(payload)

    def get_ticker(self, markets: Sequence[str]) -> pd.DataFrame:
        """Fetch the latest ticker snapshot for one or more markets."""

        params = {"markets": ",".join(markets)}
        payload = self._request("/v1/ticker", params=params)
        return pd.DataFrame(payload)

    def get_recent_trades(
        self, market: str, count: int = 100, to: Optional[datetime | str] = None
    ) -> pd.DataFrame:
        """Retrieve the most recent executed trades for a market."""

        params = {"market": market, "count": count}
        formatted_to = _format_to_param(to)
        if formatted_to:
            params["to"] = formatted_to
        payload = self._request("/v1/trades/ticks", params=params)
        df = pd.DataFrame(payload)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def get_orderbook(self, markets: Sequence[str], depth: Optional[int] = None) -> pd.DataFrame:
        """Retrieve the order book snapshot for the requested markets."""

        params = {"markets": ",".join(markets)}
        if depth is not None:
            params["count"] = depth
        payload = self._request("/v1/orderbook", params=params)
        df = pd.json_normalize(payload, record_path="orderbook_units", meta=["market"])
        return df

    def get_candles(
        self,
        market: str,
        interval: Interval = "minutes",
        unit: Optional[int] = 1,
        count: int = 200,
        to: Optional[datetime | str] = None,
        converting_price_unit: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download OHLCV candles for the specified market."""

        interval = interval.lower()
        if interval not in {"seconds", "minutes", "days", "weeks", "months", "years"}:
            raise ValueError(f"Unsupported interval '{interval}'")

        if interval == "minutes":
            if unit is None:
                raise ValueError("`unit` must be provided for minute candles")
            path = f"/v1/candles/minutes/{unit}"
        elif interval == "seconds":
            path = "/v1/candles/seconds"
        else:
            path = f"/v1/candles/{interval}"

        params = {"market": market, "count": count}
        formatted_to = _format_to_param(to)
        if formatted_to:
            params["to"] = formatted_to
        if converting_price_unit:
            params["convertingPriceUnit"] = converting_price_unit

        payload = self._request(path, params=params)
        return pd.DataFrame(payload)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def fetch_ohlcv(
        self,
        market: str,
        interval: Interval = "minutes",
        unit: Optional[int] = 1,
        count: int = 200,
        to: Optional[datetime | str] = None,
        converting_price_unit: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return candles with canonical OHLCV columns.

        The dataframe is indexed by ``(market, timestamp)`` where the
        timestamp is expressed in UTC.
        """

        candles = self.get_candles(
            market=market,
            interval=interval,
            unit=unit,
            count=count,
            to=to,
            converting_price_unit=converting_price_unit,
        )

        if candles.empty:
            return pd.DataFrame(
                columns=["market", "timestamp", "open", "high", "low", "close", "volume", "turnover"],
            ).set_index(["market", "timestamp"])

        candles = candles.copy()
        candles["timestamp"] = pd.to_datetime(candles["timestamp"], unit="ms", utc=True)
        renamed = candles.rename(
            columns={
                "opening_price": "open",
                "high_price": "high",
                "low_price": "low",
                "trade_price": "close",
                "candle_acc_trade_volume": "volume",
                "candle_acc_trade_price": "turnover",
            }
        )
        ordered_cols = [
            "market",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ]
        ordered = renamed[ordered_cols]
        ordered = ordered.sort_values("timestamp")
        return ordered.set_index(["market", "timestamp"])


def load_upbit_ohlcv(
    markets: Iterable[str],
    interval: Interval = "minutes",
    unit: Optional[int] = 1,
    count: int = 200,
    to: Optional[datetime | str] = None,
    converting_price_unit: Optional[str] = None,
    client: Optional[UpbitClient] = None,
) -> pd.DataFrame:
    """Bulk loader for OHLCV data across several markets.

    Parameters
    ----------
    markets:
        Iterable of market symbols, e.g. ``["KRW-BTC", "KRW-ETH"]``.
    interval, unit, count, to, converting_price_unit:
        Forwarded to :meth:`UpbitClient.fetch_ohlcv`.
    client:
        Optional existing :class:`UpbitClient` instance.  Supplying a shared
        client allows the caller to reuse the underlying HTTP session.

    Returns
    -------
    pandas.DataFrame
        Multi-indexed by ``(market, timestamp)`` with standard OHLCV fields.
    """

    if client is None:
        client = UpbitClient()

    frames: List[pd.DataFrame] = []
    for market in markets:
        frame = client.fetch_ohlcv(
            market=market,
            interval=interval,
            unit=unit,
            count=count,
            to=to,
            converting_price_unit=converting_price_unit,
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["market", "timestamp", "open", "high", "low", "close", "volume", "turnover"],
        ).set_index(["market", "timestamp"])

    combined = pd.concat(frames)
    combined = combined.sort_index(level=[0, 1])
    return combined


__all__ = ["UpbitClient", "load_upbit_ohlcv"]

