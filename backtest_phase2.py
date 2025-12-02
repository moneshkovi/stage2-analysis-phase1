"""
backtest_phase2.py
-------------------

Phase 2 of the Stage 2 Analysis project implements a simple
backtesting framework to evaluate the effectiveness of the Stage 2
signal derived in ``phase1.py``.  The goal is to simulate a basic
trend-following strategy where we hold a stock only when it is in
Stage 2 and compare the resulting returns against a passive
buy‑and‑hold approach.

Key features:

* Utilises the ``fetch_stock_data`` and ``stage2_signal`` functions
  defined in ``phase1.py`` to obtain data and signals.
* Computes weekly returns for both the stock and benchmark.
* Generates a binary time series representing when the stock is in
  Stage 2.  This series is used to mask returns to simulate being
  invested only during Stage 2 periods (cash otherwise).
* Computes cumulative returns and a few basic statistics (number of
  trades, total return, CAGR, strategy vs buy‑and‑hold).  More
  sophisticated metrics (drawdown, Sharpe ratio) could be added
  later.

Example usage:

    python backtest_phase2.py AAPL MSFT TSLA --benchmark SPY

This will backtest each provided ticker against the SPY benchmark and
print a small summary table.  Note that the script expects the files
from ``phase1.py`` to be available in the Python path.

"""

from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import numpy as np
import pandas as pd

from phase1 import fetch_stock_data, stage2_signal



def compute_weekly_returns(series: pd.Series) -> pd.Series:
    """Compute weekly log returns from a price series.

    Parameters
    ----------
    series : pandas.Series
        Series of prices indexed by date.

    Returns
    -------
    pandas.Series
        Series of log returns.  NaNs at the beginning are dropped.
    """
    return np.log(series / series.shift(1)).dropna()



def build_stage2_mask(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, ma_window: int = 30) -> pd.Series:
    """Generate a boolean mask indicating Stage 2 periods.

    For each week we compute the Stage 2 signal based on data up to
    that date.  We align stock and benchmark data to common dates and
    return a series of booleans (True if Stage 2, False otherwise).

    Parameters
    ----------
    stock_df : pandas.DataFrame
        DataFrame with weekly data for the stock.
    benchmark_df : pandas.DataFrame
        DataFrame with weekly data for the benchmark.
    ma_window : int
        Moving average window length used in Stage 2 detection.

    Returns
    -------
    pandas.Series
        Boolean series indexed by date.
    """
    # Align indices to intersection
    common_idx = stock_df.index.intersection(benchmark_df.index)
    mask = pd.Series(False, index=common_idx)
    # Slide window through time computing signal up to each point
    for i, dt in enumerate(common_idx):
        sub_stock = stock_df.loc[:dt]
        sub_bench = benchmark_df.loc[:dt]
        sig, _ = stage2_signal(sub_stock, sub_bench, ma_window=ma_window)
        mask.iloc[i] = sig
    return mask



def backtest_single(stock: str, benchmark: str = "SPY", ma_window: int = 30) -> Tuple[pd.Series, pd.Series]:
    """Backtest a single stock using a Stage 2 strategy.

    Parameters
    ----------
    stock : str
        Ticker symbol for the stock.
    benchmark : str
        Benchmark symbol (index ETF).
    ma_window : int
        Moving average window for Stage 2 detection.

    Returns
    -------
    (strategy_equity, buy_hold_equity)
        Two cumulative return series (exponentiated log returns) for
        the Stage 2 strategy and a buy‑and‑hold approach, respectively.
    """
    stock_df = fetch_stock_data(stock)
    bench_df = fetch_stock_data(benchmark)

    # Align for returns
    common_idx = stock_df.index.intersection(bench_df.index)
    stock_df = stock_df.loc[common_idx]
    bench_df = bench_df.loc[common_idx]

    # Returns (log returns for additive convenience)
    stock_rets = compute_weekly_returns(stock_df["Close"])
    bench_rets = compute_weekly_returns(bench_df["Close"])

    # Make sure both series share index after dropping NaNs
    common_idx = stock_rets.index.intersection(bench_rets.index)
    stock_rets = stock_rets.loc[common_idx]
    bench_rets = bench_rets.loc[common_idx]

    # Build Stage 2 mask
    mask = build_stage2_mask(stock_df.loc[common_idx], bench_df.loc[common_idx], ma_window=ma_window)
    mask = mask.loc[common_idx]

    # Strategy returns: only earn stock return when mask True, else zero (cash)
    strategy_log_returns = stock_rets * mask.astype(int)
    # For fairness we could assume cash returns 0
    strategy_equity = (strategy_log_returns.cumsum()).apply(np.exp)
    buy_hold_equity = (stock_rets.cumsum()).apply(np.exp)
    return strategy_equity, buy_hold_equity



def summarise_performance(equity: pd.Series) -> Tuple[float, float]:
    """Compute total return and CAGR from an equity series.

    Parameters
    ----------
    equity : pandas.Series
        Equity curve starting at 1.0.

    Returns
    -------
    (total_return, cagr)
        Total return (multiple) and compound annual growth rate.
    """
    total_return = equity.iloc[-1] - 1.0
    n_years = len(equity) / 52.0  # approximate weeks per year
    if n_years <= 0:
        return float("nan"), float("nan")
    cagr = equity.iloc[-1] ** (1 / n_years) - 1
    return total_return, cagr



def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Stage 2 strategy on one or more stocks.")
    parser.add_argument("stocks", nargs="*", help="Ticker symbols to backtest", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--benchmark", default="SPY", help="Benchmark symbol (default: SPY)")
    parser.add_argument("--ma", type=int, default=30, help="Moving average window length (default: 30)")
    args = parser.parse_args()

    results = []
    for stock in args.stocks:
        strat_eq, bh_eq = backtest_single(stock, benchmark=args.benchmark, ma_window=args.ma)
        strat_return, strat_cagr = summarise_performance(strat_eq)
        bh_return, bh_cagr = summarise_performance(bh_eq)
        results.append((stock, strat_return, strat_cagr, bh_return, bh_cagr))

    # Print summary table
    print(f"Backtest results (benchmark={args.benchmark}, MA={args.ma} weeks):")
    print(f"{'Ticker':<10}{'Strategy Return':>18}{'Strategy CAGR':>18}{'Buy&Hold Return':>18}{'Buy&Hold CAGR':>18}")
    for stock, s_ret, s_cagr, bh_ret, bh_cagr in results:
        print(f"{stock:<10}{s_ret:>18.2%}{s_cagr:>18.2%}{bh_ret:>18.2%}{bh_cagr:>18.2%}")



if __name__ == "__main__":
    main()
