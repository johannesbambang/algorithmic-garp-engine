import os
import matplotlib
if os.environ.get('CI'):
    matplotlib.use('Agg')

import argparse
import yaml
import sys

# Custom modules
from src.data_engine import fetch_and_cluster
from src.models import detect_macro_regime, train_prediction_model
from src.optimizer import optimize_portfolio
from src.visualizer import run_backtest_report, plot_clusters

def load_config(config_path="config.yaml"):
    """Loads the external configuration file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: {config_path} not found.")
        sys.exit(1)

def run_pipeline(config, is_test=False):
    """Executes the 5‑phase quantitative pipeline."""
    watchlist = config['test_settings']['test_watchlist'] if is_test else config['market_settings']['watchlist']
    benchmark = config['market_settings']['benchmark']
    lookback = config['test_settings']['test_period'] if is_test else config['market_settings']['lookback_period']

    print(f"🚀 Launching {'TEST' if is_test else 'PRODUCTION'} ENGINE...")
    print(f"Targeting: {watchlist} (Benchmark: {benchmark})")

    # Phase 1: Market regime
    macro_state = detect_macro_regime(benchmark)

    # Phase 2: Asset clustering
    df_results = fetch_and_cluster(watchlist, lookback_period=lookback)

    # Phase 3: Train XGBoost on the first ticker
    sample_ticker = watchlist[0]
    train_prediction_model(sample_ticker)

    # Phase 4: Portfolio optimisation
    final_weights = optimize_portfolio(watchlist, lookback=lookback)

    # Phase 5: Charts & backtest
    plot_clusters(df_results)
    run_backtest_report(watchlist, final_weights, benchmark)

    print("✅ Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithmic GARP Engine V2")
    parser.add_argument("--test-mode", action="store_true", help="Quick validation with minimal data")
    args = parser.parse_args()

    config_data = load_config()
    run_pipeline(config_data, is_test=args.test_mode)