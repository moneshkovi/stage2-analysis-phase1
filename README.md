# Stage 2 Analysis Project

This repository contains a multi‑phase project inspired by Stan Weinstein’s
**Stage Analysis**.  The objective is to build an automated pipeline
that can identify stocks entering **Stage 2** (the advancing phase),
evaluate a basic strategy based on those signals, and ultimately train
a machine‑learning model to recognise Stage 2 patterns.

## Phases

### Phase 1 – Data retrieval and rule setup

The goal of Phase 1 is to establish the data‑fetching and rule
calculation infrastructure.  The script [`phase1.py`](phase1.py)
downloads weekly OHLCV data using [`yfinance`](https://pypi.org/project/yfinance/),
computes the 30‑week simple moving average, derives a relative
strength (RS) line versus a benchmark, and applies simplified
Stage 2 conditions:

* Price is above a rising 30‑week moving average.
* The slope of the moving average is positive.
* The RS line is rising.

When run as a script it scans a universe of tickers and reports those
currently meeting the Stage 2 criteria.

### Phase 2 – Backtesting

Phase 2 introduces a simple backtesting framework implemented in
[`backtest_phase2.py`](backtest_phase2.py).  It simulates a
trend‑following strategy where the portfolio holds a stock only when
the Stage 2 signal is active (cash otherwise) and compares the
results against a buy‑and‑hold benchmark.  The backtester computes
weekly log returns, generates a Stage 2 mask over time and reports
summary statistics such as total return and CAGR.

### Phase 3 – Machine learning model (coming soon)

In the final phase we plan to label historical data using the rules
defined in Phase 1 and train a supervised classifier (e.g. Random
Forest or Gradient Boosting) to recognise Stage 2 patterns.  The
trained model could then be used to score large universes of stocks
without explicitly calculating moving averages and RS for each.

## Dependencies

See `requirements.txt` for a list of Python dependencies.  To set up
your environment you can run:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

* **Scan for Stage 2 candidates**:

  ```sh
  python phase1.py AAPL MSFT NVDA --benchmark SPY
  ```

* **Backtest Stage 2 strategy**:

  ```sh
  python backtest_phase2.py AAPL MSFT NVDA --benchmark SPY --ma 30
  ```

This will print a table comparing the Stage 2 strategy performance
against a buy‑and‑hold approach for each ticker.

## Disclaimer

This project is for educational purposes only.  It does not
constitute financial advice.  Past performance is not indicative of
future results, and trading carries risk of loss.
