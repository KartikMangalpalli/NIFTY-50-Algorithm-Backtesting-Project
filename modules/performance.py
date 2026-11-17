"""
performance.py
==============
Visualisation and reporting module for the NIFTY 50 backtesting project.

All plotting functions return a matplotlib Figure object so callers can
either display inline (Jupyter) or save to disk.

Convention
----------
- Returns are daily simple returns (not log).
- Equity curves are ₹-denominated.
- Benchmark series is the NIFTY 50 buy-and-hold at the same starting capital.
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Consistent style across all notebooks
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "lines.linewidth":   1.4,
    "font.size":         11,
})

STRATEGY_COLOR   = "#58a6ff"   # blue
BENCHMARK_COLOR  = "#f0883e"   # orange
DRAWDOWN_COLOR   = "#ff7b72"   # red
SIGNAL_BUY_COLOR = "#3fb950"   # green
SIGNAL_SELL_COLOR= "#ff7b72"   # red


# ---------------------------------------------------------------------------
# 1. Equity curve
# ---------------------------------------------------------------------------

def plot_equity_curve(
    strategy_equity: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    strategy_label: str = "Strategy",
    benchmark_label: str = "NIFTY 50 Buy & Hold",
) -> plt.Figure:
    """Plot the strategy equity curve against the benchmark.

    Parameters
    ----------
    strategy_equity : pd.Series
        Absolute ₹ portfolio value indexed by date.
    benchmark_equity : pd.Series, optional
        Buy-and-hold portfolio value (same starting capital).
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(strategy_equity.index, strategy_equity.values,
            color=STRATEGY_COLOR, label=strategy_label, linewidth=1.5)

    if benchmark_equity is not None:
        ax.plot(benchmark_equity.index, benchmark_equity.values,
                color=BENCHMARK_COLOR, label=benchmark_label,
                linewidth=1.5, linestyle="--", alpha=0.85)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Drawdown chart
# ---------------------------------------------------------------------------

def plot_drawdown(
    returns: pd.Series,
    title: str = "Drawdown",
    label: str = "Strategy Drawdown",
) -> plt.Figure:
    """Plot the percentage drawdown from peak."""
    cum      = (1 + returns.dropna()).cumprod()
    peak     = cum.cummax()
    drawdown = (cum - peak) / peak * 100   # in %

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color=DRAWDOWN_COLOR, alpha=0.6, label=label)
    ax.plot(drawdown.index, drawdown.values, color=DRAWDOWN_COLOR, linewidth=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Equity + Drawdown combined (2-panel)
# ---------------------------------------------------------------------------

def plot_equity_and_drawdown(
    strategy_equity: pd.Series,
    strategy_returns: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    title: str = "Strategy Performance",
    strategy_label: str = "Strategy",
    benchmark_label: str = "NIFTY 50 Buy & Hold",
) -> plt.Figure:
    """Two-panel plot: equity curve (top) and drawdown (bottom)."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    # --- Top: equity ---
    ax1.plot(strategy_equity.index, strategy_equity.values,
             color=STRATEGY_COLOR, label=strategy_label, linewidth=1.5)
    if benchmark_equity is not None:
        ax1.plot(benchmark_equity.index, benchmark_equity.values,
                 color=BENCHMARK_COLOR, label=benchmark_label,
                 linewidth=1.5, linestyle="--", alpha=0.85)
    ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Portfolio Value (₹)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax1.legend()
    ax1.grid(True)

    # --- Bottom: drawdown ---
    cum  = (1 + strategy_returns.dropna()).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, color=DRAWDOWN_COLOR, alpha=0.5)
    ax2.plot(dd.index, dd.values, color=DRAWDOWN_COLOR, linewidth=0.7)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.grid(True)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Metrics table
# ---------------------------------------------------------------------------

def print_metrics_table(
    metrics: dict,
    title: str = "Performance Metrics",
    benchmark_metrics: Optional[dict] = None,
) -> None:
    """Print a formatted comparison table of strategy vs benchmark metrics."""
    rows = [
        ("Total Return (%)",       "total_return_pct"),
        ("CAGR (%)",               "cagr_pct"),
        ("Ann. Volatility (%)",    "ann_volatility_pct"),
        ("Sharpe Ratio",           "sharpe_ratio"),
        ("Sortino Ratio",          "sortino_ratio"),
        ("Calmar Ratio",           "calmar_ratio"),
        ("Max Drawdown (%)",       "max_drawdown_pct"),
        ("Avg DD Duration (days)", "avg_dd_duration_days"),
        ("Longest DD (days)",      "longest_dd_days"),
        ("Win Rate (%)",           "win_rate_pct"),
        ("Avg Win (%)",            "avg_win_pct"),
        ("Avg Loss (%)",           "avg_loss_pct"),
        ("Profit Factor",          "profit_factor"),
        ("VaR 95% (%)",            "var_95_pct"),
        ("CVaR 95% (%)",           "cvar_95_pct"),
        ("Skewness",               "skewness"),
        ("Kurtosis",               "kurtosis"),
        ("Alpha (ann, %)",         "alpha_ann_pct"),
        ("Beta",                   "beta"),
        ("Information Ratio",      "info_ratio"),
    ]

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    if benchmark_metrics:
        print(f"{'Metric':<30} {'Strategy':>14} {'Benchmark':>14}")
        print("-" * 60)
        for label, key in rows:
            sv = metrics.get(key, "N/A")
            bv = benchmark_metrics.get(key, "N/A")
            sv_str = f"{sv:.3f}" if isinstance(sv, float) else str(sv)
            bv_str = f"{bv:.3f}" if isinstance(bv, float) else str(bv)
            print(f"  {label:<28} {sv_str:>14} {bv_str:>14}")
    else:
        print(f"{'Metric':<35} {'Value':>14}")
        print("-" * 50)
        for label, key in rows:
            sv = metrics.get(key, "N/A")
            sv_str = f"{sv:.3f}" if isinstance(sv, float) else str(sv)
            print(f"  {label:<33} {sv_str:>14}")
    print("=" * 60)


def metrics_to_dataframe(
    metrics_dict: dict,
    benchmark_dict: Optional[dict] = None,
    strategy_name: str = "Strategy",
) -> pd.DataFrame:
    """Convert metrics dict(s) to a styled DataFrame for Jupyter display."""
    rows = {
        "Total Return (%)":       "total_return_pct",
        "CAGR (%)":               "cagr_pct",
        "Ann. Volatility (%)":    "ann_volatility_pct",
        "Sharpe Ratio":           "sharpe_ratio",
        "Sortino Ratio":          "sortino_ratio",
        "Calmar Ratio":           "calmar_ratio",
        "Max Drawdown (%)":       "max_drawdown_pct",
        "Avg DD Duration (days)": "avg_dd_duration_days",
        "Longest DD (days)":      "longest_dd_days",
        "Win Rate (%)":           "win_rate_pct",
        "Profit Factor":          "profit_factor",
        "VaR 95% (%)":            "var_95_pct",
        "CVaR 95% (%)":           "cvar_95_pct",
        "Skewness":               "skewness",
        "Kurtosis":               "kurtosis",
        "Alpha (ann, %)":         "alpha_ann_pct",
        "Beta":                   "beta",
        "Information Ratio":      "info_ratio",
    }

    data = {strategy_name: {label: metrics_dict.get(key, np.nan)
                             for label, key in rows.items()}}
    if benchmark_dict:
        data["NIFTY 50 B&H"] = {label: benchmark_dict.get(key, np.nan)
                                 for label, key in rows.items()}

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 5. Rolling metrics
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.065,
    title: str = "Rolling Sharpe Ratio (1-Year Window)",
) -> plt.Figure:
    """Plot the rolling annualised Sharpe ratio."""
    daily_rf  = (1 + risk_free_rate) ** (1 / 252) - 1
    excess    = returns - daily_rf
    roll_mean = excess.rolling(window, min_periods=window).mean()
    roll_std  = excess.rolling(window, min_periods=window).std(ddof=1)
    roll_sharpe = roll_mean / roll_std * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(roll_sharpe.index, roll_sharpe.values, color=STRATEGY_COLOR, linewidth=1.3)
    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.axhline(1, color=SIGNAL_BUY_COLOR, linewidth=0.8, linestyle=":", label="Sharpe = 1")
    ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                    where=roll_sharpe.values >= 0, alpha=0.2, color=SIGNAL_BUY_COLOR)
    ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                    where=roll_sharpe.values <  0, alpha=0.2, color=DRAWDOWN_COLOR)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Monthly returns heatmap
# ---------------------------------------------------------------------------

def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns (%)",
) -> plt.Figure:
    """Plot a year × month heatmap of monthly returns."""
    monthly = (1 + returns).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")
    monthly_pct = monthly * 100

    pivot = monthly_pct.to_frame("ret")
    pivot["year"]  = pivot.index.year
    pivot["month"] = pivot.index.month
    heatmap_data = pivot.pivot_table(values="ret", index="year", columns="month")
    heatmap_data.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, len(heatmap_data) * 0.55 + 1.5))
    vmax = max(abs(heatmap_data.values[~np.isnan(heatmap_data.values)]).max(), 0.01)

    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(heatmap_data.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels(heatmap_data.index)

    for i in range(len(heatmap_data)):
        for j in range(12):
            val = heatmap_data.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < vmax * 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Return (%)")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Return distribution
# ---------------------------------------------------------------------------

def plot_return_distribution(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Daily Return Distribution",
) -> plt.Figure:
    """Histogram of daily returns with optional benchmark overlay."""
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = 80

    ax.hist(returns.dropna(), bins=bins, density=True, alpha=0.65,
            color=STRATEGY_COLOR, label="Strategy", edgecolor="none")

    if benchmark_returns is not None:
        ax.hist(benchmark_returns.dropna(), bins=bins, density=True, alpha=0.4,
                color=BENCHMARK_COLOR, label="NIFTY B&H", edgecolor="none")

    # Normal distribution overlay
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="#e3b341", linewidth=1.5,
            linestyle="--", label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})")

    ax.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Signal overlay on price chart
# ---------------------------------------------------------------------------

def plot_price_with_signals(
    df: pd.DataFrame,
    signal: pd.Series,
    title: str = "Price with Buy/Sell Signals",
    price_col: str = "CLOSE",
    max_points: int = 500,
) -> plt.Figure:
    """Plot price series with entry (▲) and exit (▼) signal markers.

    Parameters
    ----------
    max_points : int
        Subsample data to at most this many points to keep the chart readable.
    """
    # Subsample if too long
    step = max(1, len(df) // max_points)
    df_plot   = df.iloc[::step]
    sig_full  = signal.reindex(df.index)

    # Entry = signal goes 0 → 1; exit = signal goes 1 → 0
    entries = sig_full[(sig_full == 1) & (sig_full.shift(1) != 1)]
    exits   = sig_full[(sig_full == 0) & (sig_full.shift(1) == 1)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_plot.index, df_plot[price_col].values,
            color="#c9d1d9", linewidth=0.9, label=price_col)

    ax.scatter(entries.index, df.loc[entries.index, price_col],
               marker="^", color=SIGNAL_BUY_COLOR, s=60, zorder=5, label="Entry")
    ax.scatter(exits.index, df.loc[exits.index, price_col],
               marker="v", color=SIGNAL_SELL_COLOR, s=60, zorder=5, label="Exit")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(price_col)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig
