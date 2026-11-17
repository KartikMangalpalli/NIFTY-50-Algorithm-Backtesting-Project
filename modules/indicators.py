"""
indicators.py
=============
Pure-function technical indicator library for the NIFTY 50 backtesting project.

Design rules
------------
- Every function takes a pd.Series (or pd.DataFrame) and returns a pd.Series
  (or pd.DataFrame) aligned on the same DatetimeIndex.
- Indicators use only past and current data – no look-ahead.
- NaN is propagated correctly during the warm-up period rather than filled
  with invented values.
- No global state; all parameters are explicit keyword arguments.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Trend / Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (Wilder-compatible via adjust=False)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average (linearly decreasing weights)."""
    weights = np.arange(1, period + 1, dtype=float)
    def _wma(x):
        if len(x) < period or np.isnan(x).any():
            return np.nan
        return np.dot(x, weights) / weights.sum()
    return series.rolling(window=period).apply(_wma, raw=True)


def dema(series: pd.Series, period: int) -> pd.Series:
    """Double Exponential Moving Average: 2*EMA(x) - EMA(EMA(x))."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


def tema(series: pd.Series, period: int) -> pd.Series:
    """Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def hull_ma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half = max(1, period // 2)
    sqrt_p = max(1, int(np.sqrt(period)))
    return wma(2 * wma(series, half) - wma(series, period), sqrt_p)


# ---------------------------------------------------------------------------
# Momentum / oscillators
# ---------------------------------------------------------------------------

def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of Change: (close_t / close_{t-n}) - 1."""
    return series / series.shift(period) - 1


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI.

    Returns a Series in [0, 100].
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD indicator.

    Returns
    -------
    pd.DataFrame with columns: macd_line, signal_line, histogram
    """
    fast_ema   = ema(series, fast)
    slow_ema   = ema(series, slow)
    macd_line  = fast_ema - slow_ema
    sig_line   = ema(macd_line, signal)
    histogram  = macd_line - sig_line
    return pd.DataFrame(
        {"macd_line": macd_line, "signal_line": sig_line, "histogram": histogram},
        index=series.index,
    )


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D).

    Returns
    -------
    pd.DataFrame with columns: pct_k, pct_d
    """
    lowest_low   = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    range_hl     = highest_high - lowest_low
    pct_k = 100 * (close - lowest_low) / range_hl.replace(0, np.nan)
    pct_d = pct_k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"pct_k": pct_k, "pct_d": pct_d}, index=close.index)


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    constant: float = 0.015,
) -> pd.Series:
    """Commodity Channel Index."""
    tp   = (high + low + close) / 3
    sma_ = tp.rolling(period, min_periods=period).mean()
    mad  = tp.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma_) / (constant * mad)


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R."""
    hh = high.rolling(period, min_periods=period).max()
    ll  = low.rolling(period, min_periods=period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns
    -------
    pd.DataFrame with columns: middle, upper, lower, bandwidth, percent_b
    """
    middle = sma(series, period)
    std    = series.rolling(period, min_periods=period).std(ddof=1)
    upper  = middle + n_std * std
    lower  = middle - n_std * std
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
        },
        index=series.index,
    )


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """Keltner Channels.

    Returns
    -------
    pd.DataFrame with columns: middle, upper, lower
    """
    middle = ema(close, ema_period)
    atr_   = atr(high, low, close, atr_period)
    upper  = middle + multiplier * atr_
    lower  = middle - multiplier * atr_
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower},
                        index=close.index)


def historical_volatility(series: pd.Series, period: int = 21) -> pd.Series:
    """Annualised historical volatility (rolling std of log returns * sqrt(252))."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(period, min_periods=period).std(ddof=1) * np.sqrt(252)


# ---------------------------------------------------------------------------
# Trend identification
# ---------------------------------------------------------------------------

def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    """Donchian Channel.

    Returns
    -------
    pd.DataFrame with columns: upper, lower, middle
    """
    upper  = high.rolling(period, min_periods=period).max()
    lower  = low.rolling(period,  min_periods=period).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({"upper": upper, "lower": lower, "middle": middle},
                        index=high.index)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Average Directional Index (ADX) with +DI and -DI.

    Returns
    -------
    pd.DataFrame with columns: adx, plus_di, minus_di
    """
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    plus_dm  = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    # When both are positive, keep the larger; zero the other
    both_pos  = (plus_dm > 0) & (minus_dm > 0)
    cond_plus = both_pos & (plus_dm <= minus_dm)
    cond_minus= both_pos & (minus_dm < plus_dm)
    plus_dm[cond_plus]  = 0
    minus_dm[cond_minus] = 0

    smooth_tr   = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    smooth_pdm  = plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    smooth_mdm  = minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    plus_di  = 100 * smooth_pdm / smooth_tr.replace(0, np.nan)
    minus_di = 100 * smooth_mdm / smooth_tr.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    return pd.DataFrame(
        {"adx": adx_, "plus_di": plus_di, "minus_di": minus_di},
        index=close.index,
    )


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 7,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Supertrend indicator.

    Returns
    -------
    pd.DataFrame with columns: supertrend, direction (1 = bullish, -1 = bearish)
    """
    atr_ = atr(high, low, close, period)
    hl2  = (high + low) / 2

    basic_upper = hl2 + multiplier * atr_
    basic_lower = hl2 - multiplier * atr_

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend_ = pd.Series(np.nan, index=close.index)
    direction   = pd.Series(0, index=close.index)

    for i in range(1, len(close)):
        # Upper band
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Lower band
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Direction
        if pd.isna(supertrend_.iloc[i - 1]):
            direction.iloc[i] = 1
        elif supertrend_.iloc[i - 1] == final_upper.iloc[i - 1]:
            direction.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1
        else:
            direction.iloc[i] = -1 if close.iloc[i] < final_lower.iloc[i] else 1

        supertrend_.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return pd.DataFrame({"supertrend": supertrend_, "direction": direction},
                        index=close.index)


# ---------------------------------------------------------------------------
# Volume-based
# ---------------------------------------------------------------------------

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    sign = np.sign(close.diff())
    sign.iloc[0] = 0
    return (sign * volume).cumsum()


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Rolling VWAP over `period` trading days."""
    tp = (high + low + close) / 3
    return (tp * volume).rolling(period, min_periods=period).sum() / \
           volume.rolling(period, min_periods=period).sum()


# ---------------------------------------------------------------------------
# Mean-reversion helpers
# ---------------------------------------------------------------------------

def zscore(series: pd.Series, period: int) -> pd.Series:
    """Rolling z-score: (x - mean) / std over `period` lookback."""
    mu  = series.rolling(period, min_periods=period).mean()
    sig = series.rolling(period, min_periods=period).std(ddof=1)
    return (series - mu) / sig.replace(0, np.nan)


def spread_zscore(s1: pd.Series, s2: pd.Series, period: int) -> pd.Series:
    """Z-score of the spread (s1 - s2)."""
    return zscore(s1 - s2, period)


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------

def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 100) -> float:
    """Estimate the Hurst exponent using rescaled range (R/S) analysis.

    H < 0.5  → mean-reverting
    H ≈ 0.5  → random walk
    H > 0.5  → trending

    Parameters
    ----------
    series : pd.Series
        Price or return series (must have no NaN).
    """
    ts = series.dropna().values.astype(float)
    lags = range(min_lag, min(max_lag, len(ts) // 2))
    tau = []
    for lag in lags:
        sub = ts[: lag]
        if len(sub) < 2:
            continue
        tau.append(np.sqrt(np.std(sub, ddof=1)))
    lags_v = list(lags)[: len(tau)]
    if len(lags_v) < 2:
        return np.nan
    poly = np.polyfit(np.log(lags_v), np.log(tau), 1)
    return poly[0]


def rolling_hurst(series: pd.Series, window: int = 252, **kwargs) -> pd.Series:
    """Rolling Hurst exponent over `window` days."""
    return series.rolling(window, min_periods=window).apply(
        lambda x: hurst_exponent(pd.Series(x), **kwargs), raw=False
    )
