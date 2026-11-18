# NIFTY 50 Algorithmic Trading — Backtesting Research Project

A complete quantitative research project implementing five independent trading strategies plus an intelligent adaptive strategy selection system, backtested on ten years of NIFTY 50 index data (January 2016 – January 2026).

---

## Dataset

| Property | Value |
|---|---|
| Index | NIFTY 50 |
| File | `data/NIFTY_50_Historical_01012016_to_01012026.xlsx` |
| Period | 2016-01-01 → 2026-01-01 |
| Trading days | 2,478 |
| Columns | DATE, OPEN, HIGH, LOW, CLOSE, SHARES TRADED, TURNOVER (INR CR) |
| Missing values | None |

---

## Project Structure

```
NIFTY50_Backtesting/
│
├── data/
│   └── NIFTY_50_Historical_01012016_to_01012026.xlsx
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py      Load, validate, split and describe the dataset
│   ├── indicators.py       20+ technical indicators (SMA, EMA, RSI, MACD,
│   │                       Bollinger Bands, ATR, ADX, Donchian, z-score…)
│   ├── backtester.py       Event-driven backtester and 20 performance metrics
│   └── performance.py      Charts: equity curves, drawdown, heatmaps,
│                           return distributions, rolling Sharpe
│
├── 01_trend_following_momentum.ipynb
├── 02_mean_reversion.ipynb
├── 03_momentum.ipynb
├── 04_breakout.ipynb
├── 05_ma_crossover.ipynb
├── 06_adaptive_strategy_selection.ipynb
│
├── requirements.txt
└── README.md
```

---

## Notebooks

All six notebooks contain saved execution outputs (graphs, tables, metrics) and can be viewed without running any code.

| # | Notebook | Strategy |
|---|---|---|
| 1 | `01_trend_following_momentum.ipynb` | 12-month momentum + SMA(200) + ADX trend filter |
| 2 | `02_mean_reversion.ipynb` | Bollinger Band z-score + RSI oversold entries |
| 3 | `03_momentum.ipynb` | Composite multi-factor momentum (1M, 3M, 6M, 12M ROC) |
| 4 | `04_breakout.ipynb` | Donchian Channel breakout with volume and ATR stop |
| 5 | `05_ma_crossover.ipynb` | EMA crossover with MACD confirmation (3 systems compared) |
| 6 | `06_adaptive_strategy_selection.ipynb` | **Adaptive regime-based strategy selector** |

### Notebook 6 — Adaptive Strategy Selection

The sixth notebook is the integrating capstone. It:

1. Detects market regimes using ADX (trend strength), historical volatility, and the 200-day SMA.
2. Classifies each trading day as Bull Trend, Mean-Reverting, High Volatility, Bear/Volatile, or Transitional.
3. Routes to the most appropriate strategy based on the detected regime.
4. Backtests the combined adaptive system end-to-end.
5. Compares the adaptive system against all five individual strategies and the NIFTY 50 buy-and-hold benchmark.

All regime thresholds are computed from the training period only (2016–2021), preventing any data leakage into the validation or test periods.

---

## Shared Assumptions — All Strategies

| Parameter | Value |
|---|---|
| Initial capital | ₹100,000 |
| Transaction cost | 10 bps round-trip (5 bps entry + 5 bps exit) |
| Trade execution | Next trading-day open price |
| Direction | Long-only |
| Benchmark | NIFTY 50 buy-and-hold |
| Risk-free rate | 6.5% p.a. (Indian 91-day T-bill proxy) |
| Training set | 2016-01-01 → 2021-12-31 |
| Validation set | 2022-01-01 → 2023-06-30 |
| Test set | 2023-07-01 → 2026-01-01 |

---

## How to Run

### View saved outputs (no code execution needed)

Open any `.ipynb` file in Jupyter Notebook or JupyterLab. All graphs, tables, and metrics are already saved inside the notebooks.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter

```bash
jupyter notebook
```

Then open any notebook from the browser.

### Re-execute a single notebook

```bash
jupyter nbconvert --to notebook --execute --inplace 01_trend_following_momentum.ipynb
```

---

## Performance Summary (full period 2016–2026)

> All results are computed programmatically from live backtesting — no hardcoded metrics.

| Strategy | CAGR | Sharpe | Max Drawdown | Market Exposure |
|---|---|---|---|---|
| Trend Momentum | +2.1% | −0.41 | −19.4% | 54% |
| Mean Reversion | +1.4% | −1.36 | −7.6% | 5% |
| Multi-Factor Momentum | +1.1% | −0.95 | −11.1% | 19% |
| Breakout | +1.7% | −0.71 | −12.5% | 28% |
| MA Crossover | +3.8% | −0.32 | −15.5% | 38% |
| **Adaptive** | **+2.6%** | **−0.52** | **−13.6%** | 34% |
| NIFTY 50 B&H | +12.9% | +0.44 | −38.4% | 100% |

All strategies underperform buy-and-hold in CAGR because they are selective long-only systems in a decade-long bull market. Their value is in significantly reducing maximum drawdown.

---

## Research Integrity

- No look-ahead bias: all signals use only data available at the close of each trading day; trades execute at the following day's open.
- No data leakage: regime thresholds and model selection use only the training/validation periods.
- No hardcoded results: all metrics are computed live from actual backtesting.
- Limitations are documented in the conclusion section of each notebook.

---

## Module API Reference (for Web Development)

The four modules below are designed to be imported independently by a future backend. Each module has no cross-dependencies except `data_loader` → `indicators` → `backtester` → `performance`.

### `modules/data_loader.py`

| Function | Input | Output | Description |
|---|---|---|---|
| `load_data(filepath=None)` | `str` or `Path` (optional) | `pd.DataFrame` | Loads the Excel dataset, validates OHLCV integrity, adds derived columns (PREV_CLOSE, DAILY_RETURN, LOG_RETURN, TRUE_RANGE, NEXT_OPEN). Index = DatetimeIndex. |
| `split_data(df, train_end, val_end)` | `pd.DataFrame`, two date strings | `(train, val, test)` tuple of DataFrames | Chronological train/val/test split. No shuffling. |
| `normalise_prices(df, base_col)` | `pd.DataFrame`, column name | `pd.Series` | Rebase prices to 1.0 on first observation. |
| `describe_dataset(df)` | `pd.DataFrame` | prints summary | Human-readable dataset summary. |

### `modules/indicators.py`

All functions take `pd.Series` and return `pd.Series` or `pd.DataFrame`, aligned on the same index. No look-ahead — all use `min_periods=period` so NaN propagates during warm-up.

| Function | Input | Output | Description |
|---|---|---|---|
| `sma(series, period)` | close, int | `pd.Series` | Simple Moving Average |
| `ema(series, period)` | close, int | `pd.Series` | Exponential Moving Average |
| `rsi(series, period=14)` | close, int | `pd.Series` | Wilder RSI (0–100) |
| `macd(series, fast, slow, signal)` | close, 3 ints | `pd.DataFrame` | MACD line, signal line, histogram |
| `bollinger_bands(series, period, n_std)` | close, int, float | `pd.DataFrame` | middle, upper, lower, bandwidth, percent_b |
| `atr(high, low, close, period)` | 3 Series, int | `pd.Series` | Average True Range (Wilder) |
| `adx(high, low, close, period)` | 3 Series, int | `pd.DataFrame` | ADX, +DI, −DI |
| `donchian_channel(high, low, period)` | 2 Series, int | `pd.DataFrame` | upper, lower, middle |
| `zscore(series, period)` | close, int | `pd.Series` | Rolling z-score |
| `roc(series, period)` | close, int | `pd.Series` | Rate of Change |
| `historical_volatility(series, period=21)` | close, int | `pd.Series` | Annualised rolling std of log returns |

### `modules/backtester.py`

| Function / Class | Input | Output | Description |
|---|---|---|---|
| `Backtester(df, initial_capital, cost_bps, allow_short)` | DataFrame, float, float, bool | `Backtester` instance | Single-asset event-driven backtester. Executes signals at next-day open. |
| `bt.run(signal, params=None)` | `pd.Series` (1=long, 0=flat) | `BacktestResult` | Runs full backtest. Returns signal, position, equity curve, daily returns, trade log, and 20+ metrics dict. |
| `compute_metrics(returns, portfolio_val, initial_capital, risk_free_rate, benchmark_close)` | 2 Series, float, float, Series | `dict` | Standalone metrics: CAGR, Sharpe, Sortino, Calmar, max drawdown, win rate, VaR, CVaR, alpha, beta, info ratio. |
| `benchmark_metrics(df, initial_capital, risk_free_rate)` | DataFrame, float, float | `dict` | Buy-and-hold benchmark metrics. |

**`BacktestResult` fields:** `signal`, `position`, `exec_price`, `portfolio_val`, `returns`, `trade_log`, `params`, `metrics`.

### `modules/performance.py`

All plotting functions return a `matplotlib.figure.Figure` for inline display or saving to disk.

| Function | Input | Output | Description |
|---|---|---|---|
| `plot_equity_and_drawdown(strategy_equity, strategy_returns, benchmark_equity, ...)` | 2–3 Series | `Figure` | Two-panel: equity curve + drawdown |
| `plot_rolling_sharpe(returns, window, risk_free_rate, ...)` | Series, int, float | `Figure` | Rolling 1-year Sharpe ratio |
| `plot_monthly_returns_heatmap(returns, ...)` | Series | `Figure` | Year × month heatmap |
| `plot_return_distribution(returns, benchmark_returns, ...)` | 1–2 Series | `Figure` | Histogram with normal overlay |
| `print_metrics_table(metrics, title, benchmark_metrics)` | dict, str, dict | prints | Formatted comparison table |
| `metrics_to_dataframe(metrics_dict, benchmark_dict, strategy_name)` | 1–2 dicts, str | `pd.DataFrame` | Styled DataFrame for Jupyter display |
