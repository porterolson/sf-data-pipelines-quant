import polars as pl

path = "/home/amh1124/groups/grp_quant/database/development/barra_specific_returns/*.parquet"

print(pl.read_parquet(path))