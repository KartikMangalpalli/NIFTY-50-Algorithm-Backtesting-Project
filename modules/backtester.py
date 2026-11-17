"""
backtester.py
=============
Event-driven single-asset backtesting engine for the NIFTY 50 project.

Design principles
-----------------
- Signals are generated from the *close* price of day t.
- Execution happens at the *open* price of day t+1 (NEXT_OPEN).
  This prevents look-ahead bias while keeping the simulation realistic.
- Long-only trading (the NIFTY 50 index cannot be shorted directly).
- Proportional transaction costs are applied to the full trade value on
  every entry and exit.
- Position sizing uses the full available capital (fully-invested when long).
- Returns are computed on a per-unit basis so the equity curve can be
  initialised to any starting capital.

Usage
-----
    from modules.backtester import Backtester

    bt = Backtester(df, initial_capital=100_000, cost_bps=10)
    results = bt.run(signal_series)   # signal: 1=long, 0=flat
    bt.summary()
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data container for a single backtest result
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for all backtest outputs."""

    signal:        pd.Series           # Raw signal (1 / 0)
    position:      pd.Series           # Actual held position after execution
    exec_price:    pd.Series           # Execution price used for each day
    portfolio_val: pd.Series           # Equity curve (absolute ₹)
    returns:       pd.Series           # Daily strategy returns
    trade_log:     pd.DataFrame        # One row per completed trade
    params:        dict = field(default_factory=dict)

    # Filled in by Backtester.run()
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main backtesting class
# ---------------------------------------------------------------------------

class Backtester:
    """Single-asset event-driven backtester.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset as returned by data_loader.load_data().
        Must contain: OPEN, HIGH, LOW, CLOSE, NEXT_OPEN.
    initial_capital : float
        Starting portfolio value in ₹.
    cost_bps : float
        Round-trip transaction cost in basis points (e.g. 10 = 0.10%).
        Applied as half on entry and half on exit.
    allow_short : bool
        If True, signals of -1 trigger short positions (not currently used
        for NIFTY strategies in this project).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        cost_bps: float = 10.0,
        allow_short: bool = False,
    ):
        required = {"OPEN", "HIGH", "LOW", "CLOSE", "NEXT_OPEN"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        self.df              = df.copy()
        self.initial_capital = initial_capital
        self.cost_bps        = cost_bps
        self.allow_short     = allow_short
        self._one_way_cost   = cost_bps / 2 / 10_000   # half applied each leg

    # ------------------------------------------------------------------
    def run(self, signal: pd.Series, params: Optional[dict] = None) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        signal : pd.Series
            Daily signal aligned on the same index as self.df.
            Values: 1 (long) or 0 (flat). -1 (short) accepted only if
            allow_short=True.
        params : dict, optional
            Strategy parameters to store in the result for reference.

        Returns
        -------
        BacktestResult
        """
        # Align signal to data
        sig = signal.reindex(self.df.index).fillna(0)

        # ----------------------------------------------------------------
        # Execution model (no look-ahead):
        #   Signal generated at close of day t  → executed at OPEN of day t+1
        #
        # For a position that was entered at OPEN[t] and is still held at t:
        #   Daily return  = CLOSE[t] / OPEN[t] - 1   (entry day, intraday only)
        #
        # For a position held overnight from t-1 to t (entered earlier):
        #   Daily return  = CLOSE[t] / CLOSE[t-1] - 1 (full close-to-close)
        #
        # We model this correctly by tracking the reference price each day:
        #   - On the first day in a position: ref_price = OPEN[t]
        #   - On subsequent days:             ref_price = CLOSE[t-1]
        # So: daily_ret = CLOSE[t] / ref_price[t] - 1
        # ----------------------------------------------------------------

        close      = self.df["CLOSE"]
        open_      = self.df["OPEN"]
        prev_close = close.shift(1)

        # Position[t] = what we hold on day t (after executing signal[t-1] at OPEN[t])
        position = sig.shift(1).fillna(0)

        # Identify entry days (position changes from 0 → 1)
        pos_change  = position.diff().fillna(position.iloc[0])
        is_entry_day = (pos_change > 0)
        is_exit_day  = (pos_change < 0)

        # Reference price for return calculation
        ref_price = prev_close.copy()
        ref_price[is_entry_day] = open_[is_entry_day]  # entry day: use OPEN

        # Raw market return for each day we're in position
        raw_ret = close / ref_price - 1
        raw_ret = raw_ret.fillna(0)

        # Transaction costs: applied only on entry and exit days
        cost = (is_entry_day | is_exit_day).astype(float) * self._one_way_cost

        strategy_ret = position * raw_ret - cost

        exec_price = open_.copy()

        # Equity curve (multiplicative compounding)
        portfolio_val = self.initial_capital * (1 + strategy_ret).cumprod()
        portfolio_val.iloc[0] = self.initial_capital  # anchor

        # Build trade log
        trade_log = self._build_trade_log(sig, exec_price, self.df["CLOSE"])

        # Assemble result
        result = BacktestResult(
            signal=sig,
            position=position,
            exec_price=exec_price,
            portfolio_val=portfolio_val,
            returns=strategy_ret,
            trade_log=trade_log,
            params=params or {},
        )
        result.metrics = compute_metrics(
            strategy_ret, portfolio_val, self.initial_capital,
            benchmark_close=self.df["CLOSE"],
        )
        return result

    # ------------------------------------------------------------------
    def _build_trade_log(
        self,
        signal: pd.Series,
        exec_price: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """Build a DataFrame with one row per completed long trade."""
        trades = []
        in_trade = False
        entry_date = None
        entry_price = None

        for date, sig in signal.items():
            if not in_trade and sig == 1:
                in_trade    = True
                entry_date  = date
                # Actual entry price = OPEN of *next* day, stored in NEXT_OPEN of this row
                # But we have exec_price aligned as OPEN[t], so after shift(1) the entry
                # day is the first day we're IN position.
                entry_price = exec_price.get(date, np.nan)

            elif in_trade and sig != 1:
                in_trade = False
                exit_date  = date
                exit_price = exec_price.get(date, np.nan)
                if entry_price and exit_price and entry_price > 0:
                    ret = exit_price / entry_price - 1
                    trades.append({
                        "entry_date":  entry_date,
                        "exit_date":   exit_date,
                        "entry_price": round(entry_price, 2),
                        "exit_price":  round(exit_price, 2),
                        "return_pct":  round(ret * 100, 3),
                        "duration_days": (exit_date - entry_date).days,
                    })

        return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Performance metrics (standalone, reusable)
# ---------------------------------------------------------------------------

def compute_metrics(
    returns: pd.Series,
    portfolio_val: pd.Series,
    initial_capital: float,
    risk_free_rate: float = 0.065,
    benchmark_close: Optional[pd.Series] = None,
) -> dict:
    """Compute a comprehensive set of performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns (not log).
    portfolio_val : pd.Series
        Cumulative portfolio value.
    initial_capital : float
        Starting capital (used to verify total return).
    risk_free_rate : float
        Annual risk-free rate (default 6.5% – approx Indian 91-day T-bill).
    benchmark_close : pd.Series, optional
        Benchmark closing prices for beta/alpha calculation.

    Returns
    -------
    dict
        All metrics with descriptive keys.
    """
    rets = returns.dropna()
    if len(rets) == 0:
        return {}

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    # ---- Return metrics ------------------------------------------------
    total_return  = portfolio_val.iloc[-1] / initial_capital - 1
    n_years       = len(rets) / 252
    cagr          = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

    # ---- Risk metrics --------------------------------------------------
    ann_vol       = rets.std(ddof=1) * np.sqrt(252)
    sharpe        = (rets.mean() - daily_rf) / rets.std(ddof=1) * np.sqrt(252) if rets.std() else 0

    downside = rets[rets < daily_rf] - daily_rf
    sortino_denom = np.sqrt((downside ** 2).mean()) * np.sqrt(252) if len(downside) else 0
    sortino = (rets.mean() - daily_rf) / sortino_denom * np.sqrt(252) if sortino_denom else 0

    # ---- Drawdown metrics ----------------------------------------------
    cum_ret  = (1 + rets).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd   = drawdown.min()

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Average drawdown duration
    dd_periods = (drawdown < 0).astype(int)
    dd_groups  = (dd_periods.diff() != 0).cumsum()
    dd_lengths = dd_periods.groupby(dd_groups).sum()
    avg_dd_duration = dd_lengths[dd_lengths > 0].mean() if (dd_lengths > 0).any() else 0

    # Longest drawdown
    longest_dd = int(dd_lengths.max()) if len(dd_lengths) else 0

    # ---- Win/loss statistics -------------------------------------------
    winning = rets[rets > 0]
    losing  = rets[rets < 0]
    win_rate     = len(winning) / len(rets) if len(rets) else 0
    avg_win      = winning.mean() if len(winning) else 0
    avg_loss     = losing.mean()  if len(losing)  else 0
    profit_factor = abs(winning.sum() / losing.sum()) if losing.sum() != 0 else np.inf

    # ---- Tail risk -----------------------------------------------------
    var_95  = np.percentile(rets, 5)
    cvar_95 = rets[rets <= var_95].mean()
    skew    = rets.skew()
    kurt    = rets.kurtosis()

    # ---- Benchmark metrics ---------------------------------------------
    alpha = beta = info_ratio = np.nan
    if benchmark_close is not None:
        bm_ret = benchmark_close.pct_change().reindex(rets.index).dropna()
        aligned = rets.reindex(bm_ret.index).dropna()
        bm_aligned = bm_ret.reindex(aligned.index)
        if len(aligned) > 10:
            cov_mat  = np.cov(aligned, bm_aligned)
            beta     = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] != 0 else np.nan
            alpha    = (aligned.mean() - daily_rf - beta * (bm_aligned.mean() - daily_rf)) * 252
            te       = (aligned - bm_aligned).std(ddof=1) * np.sqrt(252)
            info_ratio = (aligned.mean() - bm_aligned.mean()) / te * np.sqrt(252) if te else np.nan

    return {
        # Return
        "total_return_pct":  round(total_return * 100, 2),
        "cagr_pct":          round(cagr * 100, 2),
        # Risk
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio":       round(sharpe, 3),
        "sortino_ratio":      round(sortino, 3),
        "calmar_ratio":       round(calmar, 3),
        # Drawdown
        "max_drawdown_pct":   round(max_dd * 100, 2),
        "avg_dd_duration_days": round(avg_dd_duration, 1),
        "longest_dd_days":    longest_dd,
        # Win/loss
        "win_rate_pct":       round(win_rate * 100, 2),
        "avg_win_pct":        round(avg_win * 100, 4),
        "avg_loss_pct":       round(avg_loss * 100, 4),
        "profit_factor":      round(profit_factor, 3),
        # Tail
        "var_95_pct":         round(var_95 * 100, 3),
        "cvar_95_pct":        round(cvar_95 * 100, 3),
        "skewness":           round(skew, 3),
        "kurtosis":           round(kurt, 3),
        # Benchmark
        "alpha_ann_pct":      round(alpha * 100, 3) if not np.isnan(alpha) else np.nan,
        "beta":               round(beta, 3)        if not np.isnan(beta)  else np.nan,
        "info_ratio":         round(info_ratio, 3)  if not np.isnan(info_ratio) else np.nan,
    }


def benchmark_metrics(
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.065,
) -> dict:
    """Compute buy-and-hold (CLOSE to CLOSE) metrics for the full dataset."""
    bh_ret = df["CLOSE"].pct_change().dropna()
    bh_val = initial_capital * (1 + bh_ret).cumprod()
    # prepend starting capital
    start  = pd.Series([initial_capital], index=[df.index[0]])
    bh_val = pd.concat([start, bh_val])
    return compute_metrics(bh_ret, bh_val, initial_capital, risk_free_rate)
