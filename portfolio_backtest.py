import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def download_data(tickers, period="5y"):
    """
    Downloads adjusted close prices for tickers and calculates daily returns.
    """
    data = yf.download(tickers, period=period)
    if 'Adj Close' in data.columns:
        data = data['Adj Close']
    else:
        data = data['Close']
    returns = data.pct_change().dropna()
    return data, returns


def optimize_weights(returns_window, rf_daily):
    """
    Markowitz optimization to maximize Sharpe ratio.
    Max weight per asset capped at 0.30 for diversification.
    """
    num_assets = returns_window.shape[1]
    mean_returns = returns_window.mean()
    cov_matrix = returns_window.cov()

    def objective(weights):
        port_return = np.sum(mean_returns * weights)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - rf_daily) / port_std * np.sqrt(252)
        return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 0.30) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        return np.array(initial_guess)


def run_backtest(returns_df, rf_daily, f=21, W=252, PV0=100_000, commission=0.001, slippage=0.0005):
    """
    Simulates ideal and real portfolio rebalancing strategies.

    Returns
    -------
    pv_ideal : pd.Series
        Equity curve for the frictionless ideal portfolio.
    pv_real : pd.Series
        Equity curve for the real portfolio (with commissions/slippage).
    weights_history : list of np.ndarray
        Per-day list of real portfolio weights (post-drift / post-rebalance).
    turnover_history : pd.Series
        Indexed by date; value = turnover on that day (0.0 on non-rebalancing days).
    """
    n_days = len(returns_df)
    tickers = returns_df.columns
    n_assets = len(tickers)

    pv_ideal = np.zeros(n_days + 1)
    pv_real = np.zeros(n_days + 1)
    pv_ideal[0] = PV0
    pv_real[0] = PV0

    weights_ideal = np.array([1.0 / n_assets] * n_assets)
    weights_real = np.array([1.0 / n_assets] * n_assets)

    weights_history = []
    turnover_dict = {}

    for t in range(n_days):
        r_t = returns_df.iloc[t].values
        date_t = returns_df.index[t]

        port_return_ideal = np.sum(weights_ideal * r_t)
        port_return_real = np.sum(weights_real * r_t)

        pv_ideal[t+1] = pv_ideal[t] * (1 + port_return_ideal)
        pv_real[t+1] = pv_real[t] * (1 + port_return_real)

        weights_ideal = weights_ideal * (1 + r_t) / (1 + port_return_ideal)
        weights_real = weights_real * (1 + r_t) / (1 + port_return_real)

        current_day_count = t + 1
        turnover_t = 0.0
        if current_day_count >= W and (current_day_count - W) % f == 0:
            returns_window = returns_df.iloc[t - W + 1 : t + 1]
            target_weights = optimize_weights(returns_window, rf_daily)

            weights_ideal = target_weights

            turnover_t = float(np.sum(np.abs(target_weights - weights_real)))
            costs = pv_real[t+1] * turnover_t * (commission + slippage)
            pv_real[t+1] -= costs
            weights_real = target_weights

        turnover_dict[date_t] = turnover_t
        weights_history.append(weights_real.copy())

    dates = [returns_df.index[0] - pd.Timedelta(days=1)] + list(returns_df.index)
    pv_ideal_series = pd.Series(pv_ideal, index=dates)
    pv_real_series = pd.Series(pv_real, index=dates)
    turnover_history = pd.Series(turnover_dict)

    return pv_ideal_series, pv_real_series, weights_history, turnover_history


def run_backtest_threshold(returns_df, rf_daily, W, PV0, commission, slippage, threshold=0.05):
    """
    Threshold-based rebalancing strategy (Real portfolio with costs).

    Rebalancing is triggered on day t if:
        max_i |w_actual_i - w_target_i| > threshold

    Target weights are recomputed daily using a rolling W-day window.

    Parameters
    ----------
    returns_df   : pd.DataFrame of daily returns, shape (T, n)
    rf_daily     : float, daily risk-free rate
    W            : int, lookback window in days for Markowitz optimization
    PV0          : float, initial portfolio value
    commission   : float, commission rate per unit of turnover
    slippage     : float, slippage rate per unit of turnover
    threshold    : float, max allowed weight deviation before rebalancing (default 0.05)

    Returns
    -------
    pv_threshold     : pd.Series, portfolio value over time
    turnover_history : pd.Series, turnover on each day (0 if no rebalancing)
    """
    n = returns_df.shape[1]
    dates = returns_df.index
    T = len(dates)

    w_actual = np.ones(n) / n
    pv = PV0

    pv_dict = {}
    turnover_dict = {}

    for t in range(T):
        date = dates[t]
        r_t = returns_df.iloc[t].values  # shape (n,)

        r_port = np.dot(w_actual, r_t)

        if t >= W:
            window_returns = returns_df.iloc[t - W:t]
            w_target = optimize_weights(window_returns, rf_daily)
        else:
            w_target = np.ones(n) / n

        max_deviation = np.max(np.abs(w_actual - w_target))

        if max_deviation > threshold:
            turnover = np.sum(np.abs(w_target - w_actual))
            costs = pv * turnover * (commission + slippage)
            turnover_dict[date] = turnover
            pv = pv * (1 + r_port) - costs
            w_actual = w_target.copy()
        else:
            turnover_dict[date] = 0.0
            pv = pv * (1 + r_port)
            w_new = w_actual * (1 + r_t)
            total = w_new.sum()
            if total > 0:
                w_actual = w_new / total

        pv_dict[date] = pv

    pv_threshold = pd.Series(pv_dict)
    turnover_history = pd.Series(turnover_dict)
    return pv_threshold, turnover_history


def analyze_frequencies(returns_df, rf_daily, W, PV0, commission, slippage, rf,
                         freq_list=None):
    """
    Sensitivity analysis: run backtest for each rebalancing frequency in freq_list.
    Returns a DataFrame with columns ['f', 'Sharpe_Ideal', 'Sharpe_Real'].
    """
    if freq_list is None:
        freq_list = [1, 5, 21, 63, 126, 252]

    results = []
    for f in freq_list:
        print(f"  Running backtest for f={f}...")
        pv_ideal, pv_real, _, _ = run_backtest(
            returns_df, rf_daily, f, W, PV0, commission, slippage
        )
        sharpe_ideal = calc_sharpe(pv_ideal, rf)
        sharpe_real  = calc_sharpe(pv_real,  rf)
        results.append({'f': f, 'Sharpe_Ideal': sharpe_ideal, 'Sharpe_Real': sharpe_real})

    return pd.DataFrame(results)


def optimize_hyperparameters(returns_df, rf_daily, PV0, commission, slippage, rf,
                              W_list=None, f_list=None):
    """
    2D grid search over lookback window W and rebalancing frequency f.
    Runs the Real portfolio backtest for each (W, f) combination.
    Returns a pivot table (pd.DataFrame) with W as rows, f as columns, Sharpe as values.
    """
    if W_list is None:
        W_list = [63, 126, 252]
    if f_list is None:
        f_list = [21, 63, 126, 252]

    records = []
    total = len(W_list) * len(f_list)
    done = 0
    for W in W_list:
        for f in f_list:
            done += 1
            print(f"  Grid search [{done}/{total}]: W={W}, f={f} ...")
            _, pv_real, _, _ = run_backtest(
                returns_df, rf_daily, f=f, W=W,
                PV0=PV0, commission=commission, slippage=slippage
            )
            sharpe = calc_sharpe(pv_real, rf)
            records.append({'W': W, 'f': f, 'Sharpe': sharpe})

    df = pd.DataFrame(records)
    pivot = df.pivot(index='W', columns='f', values='Sharpe')
    pivot.index.name = 'Lookback W (days)'
    pivot.columns.name = 'Rebalancing f (days)'
    return pivot


def calc_annualized_return(pv):
    L = len(pv) / 252
    return (pv.iloc[-1] / pv.iloc[0])**(1/L) - 1

def calc_sharpe(pv, rf):
    returns = pv.pct_change().dropna()
    mu_p = returns.mean() * 252
    sigma_p = returns.std() * np.sqrt(252)
    return (mu_p - rf) / sigma_p

def calc_sortino(pv, rf):
    returns = pv.pct_change().dropna()
    mu_p = returns.mean() * 252
    rf_daily = rf / 252
    downside_returns = returns[returns < rf_daily]
    sigma_p_minus = downside_returns.std() * np.sqrt(252)
    return (mu_p - rf) / sigma_p_minus

def calc_drawdown(pv):
    rolling_max = pv.cummax()
    drawdown = 1 - pv / rolling_max
    return drawdown

def calc_martin(pv, rf):
    returns = pv.pct_change().dropna()
    mu_p = returns.mean() * 252
    rd = calc_drawdown(pv)
    ulcer_index = np.sqrt(np.mean(rd**2))
    return (mu_p - rf) / ulcer_index

def calc_var(pv, level=0.05):
    returns = pv.pct_change().dropna()
    return -returns.quantile(level)


def build_metrics_table(pv_ideal, pv_real, rf):
    metrics = ["AR", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Martin Ratio (UPI)", "VaR 5%"]

    def get_row(pv):
        return [
            calc_annualized_return(pv),
            calc_sharpe(pv, rf),
            calc_sortino(pv, rf),
            calc_drawdown(pv).max(),
            calc_martin(pv, rf),
            calc_var(pv)
        ]

    df = pd.DataFrame({
        "Ideal Portfolio": get_row(pv_ideal),
        "Real Portfolio": get_row(pv_real)
    }, index=metrics)

    return df


def _savefig(fig, filename):
    """Ensure PLOTS_DIR exists and save figure there."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_equity_curves(pv_ideal, pv_real):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pv_ideal, label='Ideal Portfolio', color='blue')
    ax.plot(pv_real, label='Real Portfolio', color='orange')
    ax.set_title("Portfolio Equity Curves: Ideal vs Real")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend()
    ax.grid(True)
    return _savefig(fig, "equity_curves.png")


def plot_drawdowns(pv_ideal, pv_real):
    dd_ideal = calc_drawdown(pv_ideal)
    dd_real = calc_drawdown(pv_real)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(dd_ideal.index, -dd_ideal, 0, label='Ideal Drawdown', color='blue', alpha=0.4)
    ax.fill_between(dd_real.index, -dd_real, 0, label='Real Drawdown', color='orange', alpha=0.4)
    ax.set_title("Relative Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True)
    return _savefig(fig, "drawdown.png")


def plot_sharpe_vs_frequency(freq_df):
    """
    Line chart: X = rebalancing frequency (days), Y = Sharpe Ratio.
    Two lines: Ideal Portfolio and Real Portfolio.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq_df['f'], freq_df['Sharpe_Ideal'], marker='o', color='steelblue',
            label='Ideal Portfolio', linewidth=2)
    ax.plot(freq_df['f'], freq_df['Sharpe_Real'],  marker='s', color='darkorange',
            label='Real Portfolio',  linewidth=2)
    ax.set_xlabel("Rebalancing Frequency (trading days)", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title("Sensitivity Analysis: Sharpe Ratio vs Rebalancing Frequency", fontsize=14)
    ax.set_xticks(freq_df['f'].tolist())
    ax.set_xticklabels(['1\n(Daily)', '5\n(Weekly)', '21\n(Monthly)',
                         '63\n(Quarterly)', '126\n(Semi-ann.)', '252\n(Annual)'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _savefig(fig, "sharpe_vs_frequency.png")


def plot_turnover_history(turnover_history):
    """
    Bar chart of turnover values over time (only rebalancing days have non-zero values).
    """
    rebal_turnover = turnover_history[turnover_history > 0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(rebal_turnover.index, rebal_turnover.values, color='steelblue',
           alpha=0.7, width=10)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Turnover", fontsize=12)
    ax.set_title("Portfolio Turnover History (Rebalancing Days, f=21)", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return _savefig(fig, "turnover_history.png")


def plot_threshold_vs_calendar(pv_calendar, pv_threshold):
    """
    Equity curve comparison: calendar rebalancing (f=21, Real) vs threshold rebalancing.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pv_calendar.index, pv_calendar.values, color='darkorange',
            label='Calendar Rebalancing (f=21, Real)', linewidth=1.5)
    ax.plot(pv_threshold.index, pv_threshold.values, color='green',
            label='Threshold Rebalancing (threshold=5%, Real)', linewidth=1.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value (USD)", fontsize=12)
    ax.set_title("Equity Curves: Calendar vs Threshold-Based Rebalancing", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _savefig(fig, "threshold_vs_calendar.png")


def plot_heatmap(pivot_df):
    """
    Seaborn heatmap of Sharpe Ratio over the (W, f) grid.
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        cbar_kws={'label': 'Sharpe Ratio'}
    )
    ax.set_title("Sharpe Ratio Heatmap: Lookback Window W vs Rebalancing Frequency f\n(Real Portfolio with Costs)",
                 fontsize=13)
    ax.set_xlabel("Rebalancing Frequency f (trading days)", fontsize=11)
    ax.set_ylabel("Lookback Window W (trading days)", fontsize=11)
    plt.tight_layout()
    return _savefig(fig, "heatmap_sharpe.png")


def print_turnover_stats(label, turnover_history):
    """
    Print turnover statistics for a given strategy.

    Parameters
    ----------
    label            : str, name of the strategy (for display)
    turnover_history : pd.Series, turnover per day (0 on non-rebalancing days)
    """
    rebal_days = turnover_history[turnover_history > 0]
    total_rebal_days    = len(rebal_days)
    avg_turnover        = rebal_days.mean() if total_rebal_days > 0 else 0.0
    total_cum_turnover  = rebal_days.sum()

    print(f"\n{'='*50}")
    print(f"  Turnover Statistics: {label}")
    print(f"{'='*50}")
    print(f"  Total Rebalance Days      : {total_rebal_days}")
    print(f"  Avg Turnover per Rebal    : {avg_turnover:.4f}  ({avg_turnover*100:.2f}%)")
    print(f"  Total Cumulative Turnover : {total_cum_turnover:.4f}  ({total_cum_turnover*100:.2f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'NVDA', 'TSLA']
    RF = 0.02
    rf_daily = RF / 252

    print("Downloading data...")
    prices, returns_df = download_data(tickers, period="5y")

    print("\n[Stage 1-5] Running base backtest (f=21, W=252)...")
    pv_ideal, pv_real, weights_history, turnover_history = run_backtest(returns_df, rf_daily)

    print("Generating plots...")
    plot_equity_curves(pv_ideal, pv_real)
    plot_drawdowns(pv_ideal, pv_real)

    print("\nCalculating metrics...")
    metrics_df = build_metrics_table(pv_ideal, pv_real, RF)
    print("\nMetrics Summary:")
    print(metrics_df.to_string())

    print("\n[Stage 6] Running sensitivity analysis across rebalancing frequencies...")
    freq_list = [1, 5, 21, 63, 126, 252]
    freq_df = analyze_frequencies(
        returns_df, rf_daily, W=252, PV0=100_000,
        commission=0.001, slippage=0.0005, rf=RF,
        freq_list=freq_list
    )
    print("\nSensitivity Analysis Results:")
    print(freq_df.to_string(index=False))
    plot_sharpe_vs_frequency(freq_df)

    print("\n[Stage 7] Plotting turnover history for base case f=21...")
    plot_turnover_history(turnover_history)

    print("\n[Stage 8] Running threshold-based rebalancing (threshold=0.05)...")
    pv_threshold, turnover_threshold = run_backtest_threshold(
        returns_df, rf_daily, W=252, PV0=100_000,
        commission=0.001, slippage=0.0005, threshold=0.05
    )
    plot_threshold_vs_calendar(pv_real, pv_threshold)

    print("\nThreshold vs Calendar Comparison:")
    metrics_comparison = pd.DataFrame({
        'Calendar f=21 (Real)': {
            'AR':           calc_annualized_return(pv_real),
            'Sharpe':       calc_sharpe(pv_real, RF),
            'Sortino':      calc_sortino(pv_real, RF),
            'Max Drawdown': calc_drawdown(pv_real).max(),
            'Martin (UPI)': calc_martin(pv_real, RF),
            'VaR 5%':       calc_var(pv_real),
        },
        'Threshold 5% (Real)': {
            'AR':           calc_annualized_return(pv_threshold),
            'Sharpe':       calc_sharpe(pv_threshold, RF),
            'Sortino':      calc_sortino(pv_threshold, RF),
            'Max Drawdown': calc_drawdown(pv_threshold).max(),
            'Martin (UPI)': calc_martin(pv_threshold, RF),
            'VaR 5%':       calc_var(pv_threshold),
        },
    })
    print(metrics_comparison.to_string())

    print("\n[Stage 8] Turnover statistics for Calendar vs Threshold strategies:")
    print_turnover_stats("Calendar Rebalancing (f=21, Real)", turnover_history)
    print_turnover_stats("Threshold Rebalancing (threshold=5%, Real)", turnover_threshold)

    print("\n[Stage 8b] Running threshold-based rebalancing with threshold=0.10 (10%)...")
    pv_threshold_10, turnover_threshold_10 = run_backtest_threshold(
        returns_df, rf_daily, W=252, PV0=100_000,
        commission=0.001, slippage=0.0005, threshold=0.10
    )
    print_turnover_stats("Threshold Rebalancing (threshold=10%, Real)", turnover_threshold_10)

    print("\nThreshold 10% Performance Metrics:")
    metrics_10 = {
        'AR':           calc_annualized_return(pv_threshold_10),
        'Sharpe':       calc_sharpe(pv_threshold_10, RF),
        'Sortino':      calc_sortino(pv_threshold_10, RF),
        'Max Drawdown': calc_drawdown(pv_threshold_10).max(),
        'Martin (UPI)': calc_martin(pv_threshold_10, RF),
        'VaR 5%':       calc_var(pv_threshold_10),
    }
    for k, v in metrics_10.items():
        print(f"  {k:<20}: {v:.4f}")

    print("\nThree-Way Comparison: Calendar f=21 vs Threshold 5% vs Threshold 10%")
    metrics_3way = pd.DataFrame({
        'Calendar f=21 (Real)': {
            'AR':           calc_annualized_return(pv_real),
            'Sharpe':       calc_sharpe(pv_real, RF),
            'Sortino':      calc_sortino(pv_real, RF),
            'Max Drawdown': calc_drawdown(pv_real).max(),
            'Martin (UPI)': calc_martin(pv_real, RF),
            'VaR 5%':       calc_var(pv_real),
        },
        'Threshold 5% (Real)': {
            'AR':           calc_annualized_return(pv_threshold),
            'Sharpe':       calc_sharpe(pv_threshold, RF),
            'Sortino':      calc_sortino(pv_threshold, RF),
            'Max Drawdown': calc_drawdown(pv_threshold).max(),
            'Martin (UPI)': calc_martin(pv_threshold, RF),
            'VaR 5%':       calc_var(pv_threshold),
        },
        'Threshold 10% (Real)': {
            'AR':           calc_annualized_return(pv_threshold_10),
            'Sharpe':       calc_sharpe(pv_threshold_10, RF),
            'Sortino':      calc_sortino(pv_threshold_10, RF),
            'Max Drawdown': calc_drawdown(pv_threshold_10).max(),
            'Martin (UPI)': calc_martin(pv_threshold_10, RF),
            'VaR 5%':       calc_var(pv_threshold_10),
        },
    })
    print(metrics_3way.to_string())

    print("\n[Stage 9] Running 2D grid search over W and f (Real portfolio)...")
    pivot = optimize_hyperparameters(
        returns_df, rf_daily, PV0=100_000,
        commission=0.001, slippage=0.0005, rf=RF,
        W_list=[63, 126, 252],
        f_list=[21, 63, 126, 252]
    )
    print("\nSharpe Ratio Grid (W x f):")
    print(pivot.to_string())
    plot_heatmap(pivot)

    print(f"\nBacktest complete. All plots saved to '{PLOTS_DIR}'.")
