import polars as pl

score_path = "/home/stiten/groups/grp_quant/database/development/scores/scores.parquet"
signal_path = "/home/stiten/groups/grp_quant/database/development/signals/signals.parquet"
alpha_path = "/home/stiten/groups/grp_quant/database/development/alphas/alphas.parquet"
print("signal")
print(pl.read_parquet(signal_path).sort(['date', 'barrid']))
print("scores")
print(pl.read_parquet(score_path).sort(['date', 'barrid']))
print("alphas")
print(pl.read_parquet(alpha_path).sort(['date', 'barrid']))