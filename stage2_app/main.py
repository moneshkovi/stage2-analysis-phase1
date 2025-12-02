"""
FastAPI application for Stage 2 analysis.

This app exposes endpoints to perform scanning for Stage 2 candidates,
simple backtesting, and serves a small HTML dashboard for testing via
templates. It uses yfinance to download weekly data, calculates moving
averages, slopes, and relative strength slopes to detect Stage 2 setups,
and runs a basic backtest where a stock is held only when Stage 2 rules are
met.

To run this app locally:

    uvicorn stage2_app.main:app --reload --port 8000

Then visit http://localhost:8000 in your browser to use the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
try:
    import yfinance as yf  # type: ignore
except Exception:
    # If yfinance is unavailable, set to None so download_weekly_data can fall back
    yf = None  # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Precompute dates for dummy data so that synthetic series align across tickers and benchmark
DUMMY_DATES = pd.date_range(end=pd.Timestamp.today().normalize(), periods=104, freq='W')


@dataclass
class ScanResult:
    symbol: str
    last_close: float
    ma_value: float
    ma_slope: float
    rs_slope: float
    price_ma_ratio: float


def download_weekly_data(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Download weekly OHLCV data for a single ticker using yfinance.

    If yfinance is not installed or data cannot be fetched due to network
    restrictions, returns a dummy DataFrame with made-up data so that the
    demonstration API endpoints still return results. The dummy data simulates
    rising prices and volumes over time.
    """
    # If yfinance is unavailable, skip trying to fetch
    if yf is not None:
        try:
            df = yf.download(symbol, period=period, interval="1wk", auto_adjust=True)
            df.index = pd.to_datetime(df.index)
            # yfinance returns empty DataFrame if symbol is invalid; fallback to dummy then
            if not df.empty:
                return df
        except Exception:
            pass
    # create dummy data: 104 weeks of synthetic data using shared dates
    price_base = 50 + np.arange(len(DUMMY_DATES)) * 0.5  # linearly rising prices
    volume = np.random.randint(1_000_000, 5_000_000, size=len(DUMMY_DATES))
    df = pd.DataFrame({
        'Close': price_base,
        'High': price_base * 1.02,
        'Low': price_base * 0.98,
        'Open': price_base * 0.99,
        'Volume': volume,
    }, index=DUMMY_DATES)
    return df


def compute_stage2_signal(df: pd.DataFrame, benchmark: pd.Series, ma_window: int = 30) -> pd.DataFrame:
    """Compute moving average, slope and relative strength slope for Stage 2 detection."""
    df = df.copy()
    df["MA"] = df["Close"].rolling(ma_window).mean()
    df["MA_slope"] = df["MA"].diff()
    df["Price_MA_ratio"] = df["Close"] / df["MA"]
    rs = df["Close"] / benchmark
    rs_smooth = rs.rolling(10).mean()
    df["RS_slope"] = rs_smooth.diff()
    return df


def get_benchmark_series(benchmark_symbol: str, period: str = "5y") -> pd.Series:
    bench = download_weekly_data(benchmark_symbol, period)
    return bench["Close"]

def build_ml_dataset(tickers: List[str], benchmark_symbol: str, ma_window: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    """Construct a feature matrix and label vector for multiple tickers.

    Features include the percentage distance from the moving average,
    the slope of the moving average, the slope of the relative strength line,
    weekly price change and weekly volume change.  The label is 1 if
    simplified Stage 2 conditions (price above MA, positive MA slope and
    positive RS slope) are met, else 0.

    Returns
    -------
    (X, y) where X is the concatenated feature matrix and y is the label vector.
    """
    # download benchmark data once
    bench_df = download_weekly_data(benchmark_symbol, "10y")
    X_list: list[pd.DataFrame] = []
    y_list: list[pd.Series] = []
    for tic in tickers:
        try:
            stock_df = download_weekly_data(tic, "10y")
            # align dates with benchmark
            df = stock_df.join(bench_df[["Close"]].rename(columns={"Close": "benchmark_close"}), how="inner")
            # moving average
            df["ma"] = df["Close"].rolling(ma_window).mean()
            df["ma_slope"] = df["ma"].diff()
            # relative strength line and slope
            df["rs"] = df["Close"] / df["benchmark_close"]
            df["rs_slope"] = df["rs"].diff()
            # price and volume change
            df["price_change"] = df["Close"].pct_change()
            df["volume_change"] = df["Volume"].pct_change()
            df["price_ma_ratio"] = df["Close"] / df["ma"] - 1
            # label Stage 2
            df["stage2"] = (
                (df["Close"] > df["ma"]) &
                (df["ma_slope"] > 0) &
                (df["rs_slope"] > 0)
            ).astype(int)
            # drop NaN
            df = df.dropna()
            feature_cols = ["price_ma_ratio", "ma_slope", "rs_slope", "price_change", "volume_change"]
            X_list.append(df[feature_cols])
            y_list.append(df["stage2"])
        except Exception as exc:
            print(f"Error building ML dataset for {tic}: {exc}")
            continue
    if not X_list:
        raise ValueError("No valid data for ML dataset")
    X_concat = pd.concat(X_list, axis=0)
    y_concat = pd.concat(y_list, axis=0)
    return X_concat, y_concat


app = FastAPI(title="Stage2 Analysis API")

# Set up template and static file directories
templates = Jinja2Templates(directory="stage2_app/templates")
app.mount("/static", StaticFiles(directory="stage2_app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serve the dashboard index page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/scan")
async def api_scan(payload: Dict[str, Any]):
    """API endpoint to scan Stage 2 candidates.

    Expects JSON payload with keys:
      - tickers: list of ticker symbols (strings)
      - benchmark: benchmark symbol (string)
      - maWindow (optional): integer for moving average window (default 30)
    Returns a dictionary with a list of candidates.
    """
    tickers: List[str] = payload.get("tickers") or []
    benchmark_symbol: str = payload.get("benchmark") or "SPY"
    ma_window: int = int(payload.get("maWindow", 30))
    if not tickers:
        raise HTTPException(status_code=400, detail="tickers list is required")
    bench_series = get_benchmark_series(benchmark_symbol)
    candidates: List[Dict[str, Any]] = []
    for ticker in tickers:
        try:
            df = download_weekly_data(ticker)
            df = compute_stage2_signal(df, bench_series, ma_window)
            latest = df.iloc[-1]
            if (
                latest["Close"] > latest["MA"]
                and latest["MA_slope"] > 0
                and latest["RS_slope"] > 0
            ):
                res = ScanResult(
                    symbol=ticker,
                    last_close=round(float(latest["Close"]), 2),
                    ma_value=round(float(latest["MA"]), 2),
                    ma_slope=round(float(latest["MA_slope"]), 4),
                    rs_slope=round(float(latest["RS_slope"]), 4),
                    price_ma_ratio=round(float(latest["Price_MA_ratio"]), 4),
                )
                candidates.append(res.__dict__)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    return {"candidates": candidates}


@app.post("/api/backtest")
async def api_backtest(payload: Dict[str, Any]):
    """API endpoint for simple backtesting using Stage 2 rules.

    Expects JSON payload with keys:
      - tickers: list of ticker symbols
      - benchmark: benchmark symbol
      - maWindow (optional): moving average window
    Returns a dict of performance metrics per ticker.
    """
    tickers: List[str] = payload.get("tickers") or []
    benchmark_symbol: str = payload.get("benchmark") or "SPY"
    ma_window: int = int(payload.get("maWindow", 30))
    if not tickers:
        raise HTTPException(status_code=400, detail="tickers list is required")
    bench_series = get_benchmark_series(benchmark_symbol)
    results: Dict[str, Dict[str, float]] = {}
    for ticker in tickers:
        try:
            df = download_weekly_data(ticker)
            df = compute_stage2_signal(df, bench_series, ma_window)
            signal = (
                (df["Close"] > df["MA"]) & (df["MA_slope"] > 0) & (df["RS_slope"] > 0)
            ).astype(int)
            returns = df["Close"].pct_change().fillna(0)
            strategy_returns = returns * signal
            cumulative = (1 + strategy_returns).cumprod()
            total_return = cumulative.iloc[-1] - 1
            cagr = cumulative.iloc[-1] ** (1 / (len(cumulative) / 52)) - 1 if len(cumulative) > 0 else 0
            results[ticker] = {
                "total_return": round(float(total_return), 4),
                "cagr": round(float(cagr), 4),
            }
        except Exception as e:
            print(f"Error backtesting {ticker}: {e}")
            continue
    return results

@app.post("/api/ml/train")
async def api_ml_train(payload: Dict[str, Any]):
    """API endpoint to train a Stage 2 classification model.

    Expects JSON payload with keys:
      - tickers: list of ticker symbols
      - benchmark: benchmark symbol
      - maWindow (optional): moving average window
      - testSize (optional): float fraction of data for testing (default 0.2)

    Returns a dictionary with accuracy and a simple classification report.
    """
    tickers: List[str] = payload.get("tickers") or []
    benchmark_symbol: str = payload.get("benchmark") or "SPY"
    ma_window: int = int(payload.get("maWindow", 30))
    test_size: float = float(payload.get("testSize", 0.2))
    if not tickers:
        raise HTTPException(status_code=400, detail="tickers list is required")
    # build dataset
    try:
        X, y = build_ml_dataset(tickers, benchmark_symbol, ma_window)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # Combine X and y to ensure alignment when dropping NaNs
    train_df = X_train.copy()
    train_df["target"] = y_train
    # Replace inf and drop NaNs across features and target
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    X_train_clean = train_df.drop(columns=["target"])
    y_train_clean = train_df["target"]
    # Clean test set similarly
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()
    X_test_clean = test_df.drop(columns=["target"])
    y_test_clean = test_df["target"]
    # train
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_clean, y_train_clean)
    y_pred = clf.predict(X_test_clean)
    acc = accuracy_score(y_test_clean, y_pred) if len(y_test_clean) > 0 else 0.0
    report = classification_report(y_test_clean, y_pred, output_dict=True) if len(y_test_clean) > 0 else {}
    return {"accuracy": round(float(acc), 4), "report": report}

@app.post("/api/ml/predict")
async def api_ml_predict(payload: Dict[str, Any]):
    """API endpoint to predict Stage 2 probabilities using a trained model.

    This endpoint trains a new model on the provided tickers and then
    predicts the probability that the latest data point for each ticker is
    Stage 2. It returns a mapping from ticker to probability.

    Expects JSON payload with keys:
      - tickers: list of ticker symbols
      - benchmark: benchmark symbol
      - maWindow (optional): moving average window

    Note: This simple implementation trains a model on the provided tickers
    and then uses the last row of each ticker's feature set for prediction.
    """
    tickers: List[str] = payload.get("tickers") or []
    benchmark_symbol: str = payload.get("benchmark") or "SPY"
    ma_window: int = int(payload.get("maWindow", 30))
    if not tickers:
        raise HTTPException(status_code=400, detail="tickers list is required")
    # build dataset
    try:
        X, y = build_ml_dataset(tickers, benchmark_symbol, ma_window)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    # Clean the dataset before training
    full_df = X.copy()
    full_df["target"] = y
    full_df = full_df.replace([np.inf, -np.inf], np.nan).dropna()
    if full_df.empty:
        raise HTTPException(status_code=500, detail="No valid data for training")
    X_clean = full_df.drop(columns=["target"])
    y_clean = full_df["target"]
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_clean, y_clean)
    # prepare predictions dictionary
    predictions: Dict[str, float] = {}
    for tic in tickers:
        try:
            stock_df = download_weekly_data(tic, "10y")
            bench_df = download_weekly_data(benchmark_symbol, "10y")
            df = stock_df.join(bench_df[["Close"]].rename(columns={"Close": "benchmark_close"}), how="inner")
            df["ma"] = df["Close"].rolling(ma_window).mean()
            df["ma_slope"] = df["ma"].diff()
            df["rs"] = df["Close"] / df["benchmark_close"]
            df["rs_slope"] = df["rs"].diff()
            df["price_change"] = df["Close"].pct_change()
            df["volume_change"] = df["Volume"].pct_change()
            df["price_ma_ratio"] = df["Close"] / df["ma"] - 1
            df = df.dropna()
            feature_cols = ["price_ma_ratio", "ma_slope", "rs_slope", "price_change", "volume_change"]
            if df.empty:
                continue
            latest_features = df.iloc[-1][feature_cols].values.reshape(1, -1)
            # If the classifier was trained on a single class, predict_proba may not have two columns
            if len(clf.classes_) == 1 or (1 not in clf.classes_):
                # In this case assign 0.0 probability for Stage 2
                prob = 0.0
            else:
                # Determine index of class '1' in classes_
                class_index = list(clf.classes_).index(1)
                prob = clf.predict_proba(latest_features)[0][class_index]
            predictions[tic] = round(float(prob), 4)
        except Exception as exc:
            print(f"Error predicting for {tic}: {exc}")
            continue
    return predictions
