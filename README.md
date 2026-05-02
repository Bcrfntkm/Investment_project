# Portfolio Backtest — Optimal Investment & Risk Management

Final project for the "Optimal Investment & Risk Management" course.

The script downloads historical price data for 10 large-cap US stocks, builds a
Markowitz mean-variance portfolio, and compares several rebalancing strategies
across a range of performance metrics.

---

## Project structure

```
Investment_project/
├── portfolio_backtest.py   # main script (all logic lives here)
├── requirements.txt        # Python dependencies
├── Optimal_Investment_Project_Presentation.pdf        # approximate presentation with results
├── plots/                  # auto-created on first run; all charts saved here
│   ├── equity_curves.png
│   ├── drawdown.png
│   ├── sharpe_vs_frequency.png
│   ├── turnover_history.png
│   ├── threshold_vs_calendar.png
│   └── heatmap_sharpe.png
└── README.md
```

---

## What the script does

### Data
Downloads 5 years of daily adjusted-close prices from Yahoo Finance for:
`AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, V, PG, NVDA, TSLA`

### Portfolio optimisation
Each rebalancing event solves a **Markowitz mean-variance** problem via
`scipy.optimize.minimize` (SLSQP) to maximise the Sharpe ratio subject to:
- weights sum to 1
- each weight ∈ [0, 0.30] (max 30 % per asset)

### Stages

| Stage | Description |
|-------|-------------|
| 1–5 | Base backtest with calendar rebalancing (`f = 21` trading days, `W = 252`-day lookback). Compares **Ideal** (no costs) vs **Real** (commission 0.1 % + slippage 0.05 %) portfolios. Produces equity-curve and drawdown charts. |
| 6 | **Sensitivity analysis** — runs the backtest for six rebalancing frequencies (`f` = 1, 5, 21, 63, 126, 252 days) and plots Sharpe ratio vs frequency. |
| 7 | **Turnover history** — bar chart of portfolio turnover on each rebalancing day for the base case `f = 21`. |
| 8 | **Threshold-based rebalancing** — rebalances only when the maximum weight deviation from the Markowitz target exceeds a threshold (5 % and 10 %). Compares equity curves and metrics against calendar rebalancing. |
| 9 | **2-D grid search** — sweeps `W ∈ {63, 126, 252}` and `f ∈ {21, 63, 126, 252}` for the Real portfolio and visualises the Sharpe ratio as a heatmap. |

### Metrics reported
- **AR** — annualised return
- **Sharpe ratio** — `(μ − r_f) / σ`, annualised
- **Sortino ratio** — penalises only downside volatility
- **Max drawdown** — peak-to-trough relative decline
- **Martin ratio (UPI)** — `(μ − r_f) / Ulcer Index`
- **VaR 5 %** — 5th-percentile daily loss

---

## Quick start

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` lists only the extra dependency (`seaborn`); the rest
(`yfinance`, `pandas`, `numpy`, `scipy`, `matplotlib`) should be installed
alongside it automatically, or add them manually:

```bash
pip install yfinance pandas numpy scipy matplotlib seaborn
```

### 3. Run

```bash
python portfolio_backtest.py
```

The script prints progress and metric tables to stdout.
All six charts are saved to `Investment_project/plots/` (the folder is created
automatically if it does not exist).

---

## Configuration

All key parameters are set at the top of the `if __name__ == "__main__":` block:

| Variable | Default | Meaning |
|----------|---------|---------|
| `tickers` | 10 US stocks | Universe of assets |
| `RF` | `0.02` | Annual risk-free rate |
| `period` | `"5y"` | History length passed to `yf.download` |
| `f` (base) | `21` | Rebalancing frequency in trading days |
| `W` (base) | `252` | Lookback window for Markowitz optimisation |
| `PV0` | `100_000` | Initial portfolio value (USD) |
| `commission` | `0.001` | One-way commission rate (0.1 %) |
| `slippage` | `0.0005` | One-way slippage rate (0.05 %) |
| `threshold` | `0.05 / 0.10` | Weight-deviation threshold for threshold rebalancing |

---

## Output charts

| File | Description |
|------|-------------|
| `equity_curves.png` | Ideal vs Real portfolio value over time |
| `drawdown.png` | Relative drawdown (filled area) for both portfolios |
| `sharpe_vs_frequency.png` | Sharpe ratio as a function of rebalancing frequency |
| `turnover_history.png` | Per-rebalancing-day turnover (base case `f = 21`) |
| `threshold_vs_calendar.png` | Calendar (`f = 21`) vs threshold (5 %) equity curves |
| `heatmap_sharpe.png` | Sharpe ratio heatmap over the `(W, f)` grid |
