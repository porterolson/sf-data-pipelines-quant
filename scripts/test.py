import polars as pl
path = "/home/stiten/groups/grp_quant/database/development/assets/assets_2024.parquet"
df = pl.read_parquet(path)
print(df.columns)
