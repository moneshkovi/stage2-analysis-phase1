"""
phase1.py
----------------

This script implements the core logic for Phase 1 of the Stage 2 Analysis
project.  The goal of this phase is to set up the data‑fetching and rule
calculation infrastructure required to detect Stage 2 conditions as
described in Stan Weinstein’s Stage Analysis.  It pulls historical
price data, computes the 30‑week simple moving average (SMA), derives a
relative strength (RS) line versus a benchmark index and applies
baseline Stage 2 rules.

Functions provided:

* ``fetch_stock_data`` – download weekly OHLCV data for a ticker using
  yfinance.  Adjusts the period and interval for convenience.
* ``compute_moving_average`` – compute a simple moving average over a
  specified window length.
* ``compute_relative_strength`` – compute a ratio of a stock’s closes
  to a benchmark’s closes to produce an RS line.
* ``stage2_signal`` – apply simplified Stage 2 rules (price above
  rising 30‑week SMA and RS trending up) and return a Boolean signal.
* ``scan_universe`` – fetch data for a list of tickers and return
  those that currently meet the Stage 2 criteria.

When run as a script the module will scan a handful of example
tickers against a chosen benchmark (default: SPY) and print those
which satisfy the Stage 2 conditions.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1wk") -> pd.DataFrame:
    """Download OHLCV data for a given ticker.

    Parameters
    ----------
    ticker : str
        The symbol to download (e.g. "AAPL" or "RELIANCE.NS").  For
        Indian stocks you must append the appropriate suffix (``.NS``
        for NSE, ``.BO`` for BSE)【641857465317918†L51-L110】.
    period : str, optional
        Duration of history to download (default: ``"5y"``).  Accepts
        values like ``"max"``, ``"10y"``, etc.
    interval : str, optional
        Data interval (default: ``"1wk"`` for weekly data).  Use
        ``"1d"`` for daily or ``"1mo"`` for monthly.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date with ``Open``, ``High``, ``Low``,
        ``Close`` and ``Volume`` columns.
    """
    # yfinance returns weekly data aligned to Mondays; adjust for
    # consistency by resampling after download.  The interval
    # parameter ensures we get weekly bars directly from Yahoo when
    # supported; older data might still require resampling.
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    # Ensure column names are capitalised as expected (yfinance may
    # return lowercase names for dividends/splits).  Only keep the
    # standard OHLCV columns for analysis.
    data = data.rename(columns=str.title)[[col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in data.columns]]
    return data


def compute_moving_average(series: pd.Series, window: int = 30) -> pd.Series:
    """Compute a simple moving average over a given window.

    The moving average is computed on a rolling basis using a
    forward‑looking alignment so that the value at time ``t`` is the
    average of the previous ``window`` periods ending at ``t``.  This
    corresponds to the typical definition of a trailing moving
    average【641857465317918†L51-L110】.

    Parameters
    ----------
    series : pandas.Series
        Input time series (e.g. closing prices).
    window : int, optional
        Number of periods over which to compute the average (default: 30).

    Returns
    -------
    pandas.Series
        The moving average series aligned with the input index.
    """
    return series.rolling(window=window, min_periods=window).mean()


def compute_relative_strength(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.Series:
    """Compute a relative strength (RS) line between a stock and a benchmark.

    RS is defined as the ratio of the stock's closing price to the
    benchmark's closing price.  A rising RS line indicates the stock
    outperforming the benchmark【78118354015411†L65-L80】.  The input
    dataframes must have an overlapping date index.  If there are
    missing dates the method will align them using an inner join.

    Parameters
    ----------
    stock_df : pandas.DataFrame
        Weekly OHLCV data for the stock.
    benchmark_df : pandas.DataFrame
        Weekly OHLCV data for the benchmark (e.g. an index ETF).

    Returns
    -------
    pandas.Series
        A series of RS values indexed by date.
    """
    # Align on the intersection of index dates
    aligned = stock_df["Close"].to_frame("stock_close").join(
        benchmark_df["Close"].rename("benchmark_close"), how="inner"
    )
    return aligned["stock_close"] / aligned["benchmark_close"]


@dataclass
class Stage2Metrics:
    ticker: str
    price: float
    ma: float
    ma_slope: float
    rs: float
    rs_slope: float


def stage2_signal(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, ma_window: int = 30) -> Tuple[bool, Stage2Metrics]:
    """Determine whether a stock currently meets simplified Stage 2 criteria.

    The classical Stage 2 rules include a price above a rising 30‑week
    SMA, a breakout from a Stage 1 base and increasing volume【78118354015411†L65-L80】.  For Phase 1 we
    implement a subset of these rules:

    1. The price is above its 30‑week SMA.
    2. The 30‑week SMA is trending up (positive slope between the last
       two observations).
    3. The relative strength line vs the benchmark is rising (positive
       slope between the last two observations).

    Parameters
    ----------
    stock_df : pandas.DataFrame
        Weekly OHLCV data for the stock.
    benchmark_df : pandas.DataFrame
        Weekly OHLCV data for the benchmark.
    ma_window : int, optional
        Length of the moving average window (default: 30).

    Returns
    -------
    (bool, Stage2Metrics)
        A tuple where the first element indicates if Stage 2
        conditions are met and the second contains key metrics at the
        most recent date.
    """
    # Align on intersection to avoid NaNs at end due to mismatched dates
    rs_line = compute_relative_strength(stock_df, benchmark_df)

    # Compute moving average on stock close prices
    ma = compute_moving_average(stock_df["Close"], window=ma_window)

    # Last two points for slope calculation
    if len(ma.dropna()) < 2 or len(rs_line.dropna()) < 2:
        # Not enough history for analysis
        return False, Stage2Metrics(ticker="", price=float("nan"), ma=float("nan"), ma_slope=float("nan"), rs=float("nan"), rs_slope=float("nan"))

    # Ensure alignment by trimming to common index
    common_index = ma.dropna().index.intersection(rs_line.dropna().index)
    if len(common_index) < 2:
        return False, Stage2Metrics(ticker="", price=float("nan"), ma=float("nan"), ma_slope=float("nan"), rs=float("nan"), rs_slope=float("nan"))

    # Current and previous points
    last_idx, prev_idx = common_index[-1], common_index[-2]
    last_price = stock_df.loc[last_idx, "Close"]
    last_ma = ma.loc[last_idx]
    ma_slope = ma.loc[last_idx] - ma.loc[prev_idx]
    last_rs = rs_line.loc[last_idx]
    rs_slope = rs_line.loc[last_idx] - rs_line.loc[prev_idx]

    # Check Stage 2 criteria
    signal = (last_price > last_ma) and (ma_slope > 0) and (rs_slope > 0)

    metrics = Stage2Metrics(
        ticker="", price=float(last_price), ma=float(last_ma), ma_slope=float(ma_slope), rs=float(last_rs), rs_slope=float(rs_slope)
    )
    return signal, metrics


def scan_universe(tickers: List[str], benchmark: str = "SPY") -> List[Tuple[str, Stage2Metrics]]:
    """Scan a list of tickers and return those meeting Stage 2 conditions.

    Parameters
    ----------
    tickers : list of str
        Symbols to evaluate.  Must include appropriate exchange suffixes
        for non‑US securities.
    benchmark : str, optional
        Symbol representing the benchmark index ETF (default: ``"SPY"``).

    Returns
    -------
    list of tuples
        Each tuple contains the ticker and its Stage2Metrics if Stage 2
        conditions are met at the latest date.
    """
    results: List[Tuple[str, Stage2Metrics]] = []
    # Download benchmark once
    benchmark_df = fetch_stock_data(benchmark)
    for tic in tickers:
        stock_df = fetch_stock_data(tic)
        sig, metrics = stage2_signal(stock_df, benchmark_df)
        metrics.ticker = tic
        if sig:
            results.append((tic, metrics))
    return results


def main() -> None:
    """Run a demonstration scan on a small universe of symbols.

    Modify the ``symbols`` list to include tickers of interest.  For
    Indian stocks, append ``.NS`` (NSE) or ``.BO`` (BSE).  The
    benchmark defaults to ``SPY`` but can be changed via the
    ``--benchmark`` command‑line argument.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Scan for Stage 2 candidates based on weekly data.")
    parser.add_argument("symbols", nargs="*", help="Symbols to scan (default: AAPL, MSFT, NVDA, TSLA)", default=["AAPL", "MSFT", "NVDA", "TSLA"])
    parser.add_argument("--benchmark", default="SPY", help="Benchmark symbol (default: SPY)")
    args = parser.parse_args()
    candidates = scan_universe(args.symbols, benchmark=args.benchmark)
    if candidates:
        print("Stage 2 candidates:")
        for ticker, metrics in candidates:
            print(f"{ticker}: price={metrics.price:.2f}, MA={metrics.ma:.2f}, MA_slope={metrics.ma_slope:.2f}, RS={metrics.rs:.3f}, RS_slope={metrics.rs_slope:.3f}")
    else:
        print("No Stage 2 candidates found in the specified universe.")


if __name__ == "__main__":
    main()
