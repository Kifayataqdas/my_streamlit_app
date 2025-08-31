"""
Stock Direction Predictor — dual-mode app (Streamlit when available, CLI fallback otherwise)

Why this rewrite?
- Your sandbox throws: `ModuleNotFoundError: No module named 'streamlit'`.
- We cannot install packages here, so the app now **runs without Streamlit** using a CLI fallback.
- If you later run it on a machine that has Streamlit installed, the **same file** will launch the full dashboard.

What you get:
- Synthetic demo stock universe so nothing relies on the internet.
- Goal scanning: given an investment amount, target (amount or %), and timeframe, it ranks tickers by historical probability of hitting your goal.
- PNG charts + a CSV of recommendations (in CLI mode).
- A small test suite you can run with `--run-tests`.

Usage (in this sandbox):
    python app.py --demo --invest 200 --target-pct 100 --months 2
    python app.py --run-tests   # runs unit tests, no Streamlit required

Usage (on your own machine with Streamlit installed):
    streamlit run app.py

Notes:
- Educational only; not financial advice.
- The historical-probability method does NOT predict the future.
"""

from __future__ import annotations

import os
import sys
import argparse
import datetime as dt
import math
import textwrap
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Use a non-interactive backend so PNGs save in CLI mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Try Streamlit (optional) -------------------------------------------------
try:  # Streamlit may not be available in the sandbox
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None  # type: ignore
    STREAMLIT_AVAILABLE = False

# ---------------------------
# Synthetic demo universe
# ---------------------------

def generate_synthetic_universe(n_stocks: int = 10, n_days: int = 500, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate a small universe of synthetic OHLCV data for demo use.
    Deterministic for a given seed, no network required.
    """
    rng = np.random.default_rng(seed)
    universe: dict[str, pd.DataFrame] = {}
    dates = pd.bdate_range(end=dt.date.today(), periods=n_days)
    for i in range(n_stocks):
        drift = rng.uniform(-0.02, 0.15)  # per-step mean
        vol = rng.uniform(0.6, 2.0)       # per-step std
        close = np.cumsum(rng.normal(drift, vol, n_days)) + 50 + i * 5
        openp = close + rng.normal(0.0, vol * 0.1, n_days)
        high = np.maximum(openp, close) + np.abs(rng.normal(0.0, vol * 0.5, n_days))
        low = np.minimum(openp, close) - np.abs(rng.normal(0.0, vol * 0.5, n_days))
        volume = rng.lognormal(mean=12, sigma=0.6, size=n_days).astype(int)
        df = pd.DataFrame(
            {"Open": openp, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
        universe[f"DEMO{i+1:02d}"] = df
    return universe


def slide_window_goal_probability(close_series: pd.Series, window_days: int, required_mult: float) -> tuple[float, float, list[float]]:
    """Compute historical probability that price after `window_days` is
    >= `required_mult` * price_at_start, and the average multiplier.
    Returns: (probability, expected_multiplier, list_of_multipliers)
    """
    prices = close_series.dropna()
    n = len(prices)
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if n <= window_days:
        return 0.0, 1.0, []

    wins = 0
    multipliers: list[float] = []
    for start in range(n - window_days):
        start_price = float(prices.iloc[start])
        end_price = float(prices.iloc[start + window_days])
        mult = end_price / max(start_price, 1e-12)
        multipliers.append(mult)
        if mult >= required_mult:
            wins += 1
    prob = wins / len(multipliers) if multipliers else 0.0
    exp_mult = float(np.mean(multipliers)) if multipliers else 1.0
    return prob, exp_mult, multipliers


def scan_universe_for_goal(universe: dict[str, pd.DataFrame], window_days: int, required_mult: float) -> list[tuple[str, dict]]:
    """For each ticker dataframe, compute goal probability & stats.
    Returns a list of (ticker, info) sorted by probability desc, then exp_mult.
    """
    results: list[tuple[str, dict]] = []
    for ticker, df in universe.items():
        prob, exp_mult, multipliers = slide_window_goal_probability(df["Close"], window_days, required_mult)
        results.append((ticker, {"prob": prob, "exp_mult": exp_mult, "multipliers": multipliers}))
    results.sort(key=lambda x: (x[1]["prob"], x[1]["exp_mult"]), reverse=True)
    return results


# ---------------------------
# Plot helpers (work in CLI)
# ---------------------------

def plot_multiplier_hist(multipliers: list[float], outpath: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(multipliers, bins=40)
    ax.set_title("Distribution of window multipliers")
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ---------------------------
# Streamlit UI (only if installed)
# ---------------------------

def run_streamlit_dashboard() -> None:
    if not STREAMLIT_AVAILABLE:
        raise RuntimeError("Streamlit is not available. Run the CLI mode instead: `python app.py --demo`.")

    st.set_page_config(page_title="Trading Dashboard — Demo", layout="wide")
    st.title("📊 Trading Dashboard — Demo")

    # Sidebar inputs
    invest_amount = st.sidebar.number_input("Investment amount ($)", value=200.0, min_value=1.0)
    use_target_amount = st.sidebar.checkbox("Specify target amount", value=False)
    if use_target_amount:
        target_amount = st.sidebar.number_input("Target amount ($)", value=400.0, min_value=1.0)
        required_mult = max(target_amount / max(invest_amount, 1e-12), 1e-6)
    else:
        target_pct = st.sidebar.number_input("Target return (%)", value=100.0, min_value=1.0)
        required_mult = 1.0 + target_pct / 100.0

    timeframe_months = st.sidebar.slider("Timeframe (months)", 1, 12, 2)
    window_days = max(1, int(timeframe_months * 21))
    n_demo = st.sidebar.slider("Number of demo tickers", 4, 30, 12)
    min_prob = st.sidebar.slider("Min probability to display", 0.0, 1.0, 0.05)

    if st.button("Find matching stocks"):
        universe = generate_synthetic_universe(n_demo, n_days=800)
        results = scan_universe_for_goal(universe, window_days, required_mult)
        filtered = [(t, info) for (t, info) in results if info["prob"] >= min_prob]

        if not filtered:
            st.info("No tickers met the minimum probability threshold. Try lowering the threshold or extending the timeframe.")
            return

        # Results table
        rows = [
            {
                "Ticker": t,
                "Prob": f"{info['prob']:.2%}",
                "ExpMultiplier": f"{info['exp_mult']:.3f}",
                "Est. Final ($)": f"{invest_amount * info['exp_mult']:.2f}",
            }
            for t, info in filtered[:50]
        ]
        st.subheader("Recommended tickers (historical demo)")
        st.dataframe(pd.DataFrame(rows))

        # Top candidate details
        top_ticker, top_info = filtered[0]
        st.markdown(f"### Top candidate: **{top_ticker}** — historical P(reach goal) = {top_info['prob']:.2%}")
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(top_info["multipliers"], bins=40)
            ax.set_title("Multiplier distribution")
            st.pyplot(fig)
        with col2:
            st.metric("Estimated final value ($)", f"{invest_amount * top_info['exp_mult']:.2f}")
        st.caption("Demo data shown. Historical stats are not a guarantee of future returns.")


# ---------------------------
# CLI runner (works without Streamlit)
# ---------------------------

def run_cli(args: argparse.Namespace) -> int:
    # Resolve target multiplier
    if args.target_amount is not None:
        if args.invest <= 0:
            print("--invest must be > 0 when using --target-amount", file=sys.stderr)
            return 2
        required_mult = args.target_amount / args.invest
    else:
        pct = args.target_pct if args.target_pct is not None else 100.0
        required_mult = 1.0 + pct / 100.0

    window_days = max(1, int(args.months * 21))

    # Build demo universe
    universe = generate_synthetic_universe(n_stocks=args.n_demo, n_days=800, seed=args.seed)

    # Scan
    results = scan_universe_for_goal(universe, window_days, required_mult)
    topk = results[: args.top_k]

    # Prepare outputs
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "recommendations.csv")
    rows = [
        {
            "Ticker": t,
            "Probability": info["prob"],
            "ExpectedMultiplier": info["exp_mult"],
            "EstimatedFinal": args.invest * info["exp_mult"],
        }
        for t, info in topk
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Save a histogram for the top candidate
    if topk and topk[0][1]["multipliers"]:
        png_path = os.path.join(args.outdir, "top_multipliers.png")
        plot_multiplier_hist(topk[0][1]["multipliers"], png_path)
    else:
        png_path = None

    # Console summary
    print("\n=== Goal scan (DEMO data) ===")
    print(f"Investment: ${args.invest:.2f}  |  Target multiplier: {required_mult:.3f}  |  Window: {window_days} business days")
    print("Top results:")
    for t, info in topk:
        print(f"  {t:8s}  Prob={info['prob']:.2%}  ExpMult={info['exp_mult']:.3f}  EstFinal=${args.invest * info['exp_mult']:.2f}")
    print(f"\nSaved: {csv_path}")
    if png_path:
        print(f"Saved: {png_path}")

    return 0


# ---------------------------
# Minimal unit tests (no Streamlit needed)
# ---------------------------

def _build_df_from_series(vals: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range(end=dt.date.today(), periods=len(vals))
    return pd.DataFrame({"Close": vals, "Open": vals, "High": vals, "Low": vals, "Volume": 1}, index=idx)


def run_tests() -> int:
    import unittest

    class TestGoalScanner(unittest.TestCase):
        def test_prob_all_hits(self):
            # Strictly doubling each step: multipliers over window=1 are all 2.0
            s = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0])
            p, em, m = slide_window_goal_probability(s, window_days=1, required_mult=1.5)
            self.assertAlmostEqual(p, 1.0)
            self.assertTrue(all(mm >= 2.0 - 1e-9 for mm in m))

        def test_prob_none_hit(self):
            s = pd.Series([16.0, 8.0, 4.0, 2.0, 1.0])
            p, em, m = slide_window_goal_probability(s, window_days=1, required_mult=1.5)
            self.assertAlmostEqual(p, 0.0)
            self.assertTrue(all(mm <= 1.0 + 1e-9 for mm in m))

        def test_window_too_large(self):
            s = pd.Series([1.0, 2.0, 3.0])
            p, em, m = slide_window_goal_probability(s, window_days=10, required_mult=2.0)
            self.assertEqual(p, 0.0)
            self.assertEqual(em, 1.0)
            self.assertEqual(m, [])

        def test_scan_sorting(self):
            uni = {
                "A": _build_df_from_series([1, 2, 3, 4, 5]),  # rising
                "B": _build_df_from_series([5, 4, 3, 2, 1]),  # falling
            }
            res = scan_universe_for_goal(uni, window_days=1, required_mult=1.2)
            self.assertEqual(res[0][0], "A")  # rising series should rank first

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestGoalScanner)
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


# ---------------------------
# Entrypoint
# ---------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stock Direction Predictor — dual mode (Streamlit/CLI)")
    parser.add_argument("--ui", action="store_true", help="Force launch Streamlit dashboard (requires streamlit)")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests and exit")

    # CLI goal inputs
    parser.add_argument("--invest", type=float, default=200.0, help="Investment amount in USD")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--target-amount", type=float, help="Target amount in USD (overrides --target-pct)")
    grp.add_argument("--target-pct", type=float, help="Target return in percent (default 100%% if not provided)")
    parser.add_argument("--months", type=int, default=2, help="Timeframe in months (approx 21 business days per month)")

    # Demo universe controls
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data (default in this sandbox)")
    parser.add_argument("--n-demo", type=int, default=12, help="Number of demo tickers to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--top-k", type=int, default=5, help="How many top recommendations to print/save")
    parser.add_argument("--outdir", type=str, default="./outputs", help="Where to save CSV/PNGs in CLI mode")

    args = parser.parse_args(argv)

    if args.run_tests:
        return run_tests()

    # Decide UI vs CLI
    if args.ui and STREAMLIT_AVAILABLE:
        run_streamlit_dashboard()
        return 0
    elif args.ui and not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed; falling back to CLI. Run without --ui or install streamlit.")

    # Default to CLI in this sandbox
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
