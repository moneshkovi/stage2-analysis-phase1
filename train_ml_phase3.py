"""train_ml_phase3.py
-----------------------

This module implements Phase 3 of the Stage 2 Analysis project.  The
goal of this phase is to construct a simple machine‑learning pipeline
that can recognise when a stock is entering Stan Weinstein’s Stage 2
(the advancing phase) using historical price and volume data.  The
rules defined in ``phase1.py`` are used to create labels for the
training set (supervised learning), and a handful of engineered
features are derived from the raw OHLCV time series.  A random
forest classifier from scikit‑learn is then trained and evaluated on
a hold‑out test split.

The script exposes a ``main`` function which can be executed from
the command line.  By default it will download weekly data for a
small universe of US stocks along with a benchmark index, build a
feature/label dataset, train a model and report basic accuracy and
classification metrics.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    # Import the Phase 1 helpers from the local package
    from phase1 import fetch_stock_data, compute_moving_average, compute_relative_strength
except ImportError:
    # Allow execution when phase1 is not in the Python path
    sys.path.append(".")
    from phase1 import fetch_stock_data, compute_moving_average, compute_relative_strength  # type: ignore


def extract_features_and_labels(
    stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, ma_window: int = 30
) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute engineered features and Stage 2 labels for a single ticker.

    The following features are derived:

    * ``price_ma_ratio`` – percentage distance of the closing price from its
      moving average: ``(Close / SMA) - 1``.
    * ``ma_slope`` – first difference of the moving average (SMA[t] - SMA[t-1]).
    * ``rs_slope`` – first difference of the relative strength ratio.
    * ``price_change`` – weekly percentage change of the closing price.
    * ``volume_change`` – weekly percentage change of the volume.

    The label is a Boolean (converted to integer) indicating whether the
    simplified Stage 2 conditions defined in Phase 1 are met at that point
    in time: price above its SMA, positive SMA slope, positive RS slope.

    Parameters
    ----------
    stock_df : pandas.DataFrame
        Weekly OHLCV data for the stock with ``Close`` and ``Volume`` columns.
    benchmark_df : pandas.DataFrame
        Weekly OHLCV data for the benchmark with a ``Close`` column.
    ma_window : int, optional
        Window length for the simple moving average (default: 30).

    Returns
    -------
    (DataFrame, Series)
        A tuple containing the feature matrix and the label vector.  Both
        are indexed by date and have rows for which all features and
        labels are valid (rows with NaNs are dropped).
    """
    # Align the stock and benchmark on common dates
    df = stock_df.join(benchmark_df[["Close"]].rename(columns={"Close": "benchmark_close"}), how="inner")

    # Compute moving average
    df["ma"] = compute_moving_average(df["Close"], window=ma_window)
    # Compute slopes (differences)
    df["ma_slope"] = df["ma"].diff()

    # Relative strength line
    df["rs"] = df["Close"] / df["benchmark_close"]
    df["rs_slope"] = df["rs"].diff()

    # Price and volume dynamics
    df["price_change"] = df["Close"].pct_change()
    df["volume_change"] = df["Volume"].pct_change()
    # Distance from MA
    df["price_ma_ratio"] = df["Close"] / df["ma"] - 1

    # Label: Stage 2 conditions (price above MA, positive MA slope, positive RS slope)
    df["stage2"] = (
        (df["Close"] > df["ma"]) &
        (df["ma_slope"] > 0) &
        (df["rs_slope"] > 0)
    ).astype(int)

    # Drop rows with any NaN values to get clean feature/label set
    df = df.dropna()

    # Select features and label
    feature_cols = ["price_ma_ratio", "ma_slope", "rs_slope", "price_change", "volume_change"]
    X = df[feature_cols]
    y = df["stage2"]
    return X, y


def build_dataset(tickers: List[str], benchmark: str, ma_window: int = 30) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct a feature/label dataset across multiple tickers.

    Parameters
    ----------
    tickers : list of str
        Symbols to include in the training corpus.  Tickers should include
        exchange suffixes for non‑US securities.
    benchmark : str
        Symbol of the benchmark index ETF (e.g., ``"SPY"``).
    ma_window : int, optional
        Moving average window length (default: 30).

    Returns
    -------
    (DataFrame, Series)
        Concatenated feature matrix and label vector for all tickers.
    """
    # Fetch benchmark once
    benchmark_df = fetch_stock_data(benchmark, period="10y", interval="1wk")
    all_features: List[pd.DataFrame] = []
    all_labels: List[pd.Series] = []
    for tic in tickers:
        stock_df = fetch_stock_data(tic, period="10y", interval="1wk")
        # Skip if not enough data
        if len(stock_df) < ma_window + 2:
            continue
        X, y = extract_features_and_labels(stock_df, benchmark_df, ma_window=ma_window)
        if not X.empty:
            all_features.append(X)
            all_labels.append(y)
    if not all_features:
        raise ValueError("No valid data collected for the given tickers.")
    # Concatenate along index (dates may not align across tickers)
    X_concat = pd.concat(all_features, axis=0)
    y_concat = pd.concat(all_labels, axis=0)
    return X_concat, y_concat


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> None:
    """Train a classifier and report performance metrics.

    A random forest classifier is trained on a randomly shuffled train
    set and evaluated on the remaining hold‑out data.  Results are
    printed to stdout.  The classifier hyperparameters are kept
    conservative for simplicity.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Label vector.
    test_size : float, optional
        Fraction of the dataset to use for the test split (default: 0.2).
    random_state : int, optional
        Seed for reproducible shuffling (default: 42).
    """
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Replace any infinite values that might slip through
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
    # Align labels to cleaned features
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    # Train model
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))


def main() -> None:
    """Entry point for command‑line execution.

    Usage examples:

    ::

        python train_ml_phase3.py AAPL MSFT NVDA --benchmark SPY

    The script will download data for the specified tickers and
    benchmark, construct the feature/label dataset, train a model and
    print summary metrics.  If no tickers are provided a small
    default universe is used.
    """
    parser = argparse.ArgumentParser(description="Train a Stage 2 detection classifier using historical OHLCV data.")
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Symbols to include in the training set (default: AAPL, MSFT, NVDA, TSLA, JPM)",
        default=["AAPL", "MSFT", "NVDA", "TSLA", "JPM"],
    )
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark symbol used to compute relative strength (default: SPY)",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=30,
        help="Window length for the moving average (default: 30)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing (default: 0.2)",
    )
    args = parser.parse_args()

    # Build dataset
    print(f"Fetching data for tickers: {args.tickers} and benchmark: {args.benchmark}")
    X, y = build_dataset(args.tickers, benchmark=args.benchmark, ma_window=args.ma_window)

    # Train and evaluate
    train_and_evaluate(X, y, test_size=args.test_size)


if __name__ == "__main__":
    main()
