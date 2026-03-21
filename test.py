import polars as pl
from pipelines.signals import SIGNALS

score_path = "/home/stiten/groups/grp_quant/database/development/scores/scores.parquet"
signal_path = "/home/stiten/groups/grp_quant/database/development/signals/signals.parquet"
alpha_path = "/home/stiten/groups/grp_quant/database/development/alphas/alphas.parquet"

# Expected signals
expected_signals = set(SIGNALS.keys())

# Load tables
signals_df = pl.read_parquet(signal_path)
scores_df = pl.read_parquet(score_path)
alphas_df = pl.read_parquet(alpha_path)

print("=" * 60)
print("SIGNAL COMPLETENESS CHECK")
print("=" * 60)

# Check all signals are present
present_signals = set(signals_df["signal_name"].unique())
missing_signals = expected_signals - present_signals
extra_signals = present_signals - expected_signals

print(f"\nExpected signals: {sorted(expected_signals)}")
print(f"Present signals: {sorted(present_signals)}")

if missing_signals:
    print(f"❌ MISSING SIGNALS: {missing_signals}")
else:
    print("✓ All expected signals present")

if extra_signals:
    print(f"⚠ Extra signals found: {extra_signals}")

# Row counts per signal
print("\n" + "=" * 60)
print("ROW COUNTS BY SIGNAL")
print("=" * 60)

for table_name, df in [("signals", signals_df), ("scores", scores_df), ("alphas", alphas_df)]:
    print(f"\n{table_name.upper()}:")
    counts = df.group_by("signal_name").agg(pl.len().alias("count")).sort("signal_name")
    print(counts)

    # Check all signals have same count (shouldn't differ)
    count_values = counts["count"].to_list()
    if len(set(count_values)) > 1:
        print(f"⚠ Warning: Signal row counts vary")

# Distribution stats
print("\n" + "=" * 60)
print("SIGNAL VALUE DISTRIBUTIONS")
print("=" * 60)

for signal_name in sorted(expected_signals):
    signal_vals = signals_df.filter(pl.col("signal_name") == signal_name)["signal_value"]
    print(f"\n{signal_name}:")
    print(f"  Count: {len(signal_vals)}")
    print(f"  Nulls: {signal_vals.null_count()}")
    print(f"  Min: {signal_vals.min():.4f}")
    print(f"  Max: {signal_vals.max():.4f}")
    print(f"  Mean: {signal_vals.mean():.4f}")
    print(f"  Std: {signal_vals.std():.4f}")