import polars as pl

path = "/home/amh1124/groups/grp_quant/database/development/ftse_russell/*.parquet"

print(pl.read_parquet(path))