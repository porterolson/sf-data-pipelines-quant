from datetime import date
import polars as pl
from pipelines.utils.tables import Database
from tqdm import tqdm

def assets_backfill_flow(start_date: date, end_date: date, database: Database) -> None:
    """
    Materialize assets table by combining data from multiple sources.

    Strategy:
    1. Lazy scan all sources at once
    2. Chain all lazy joins on all data
    3. Collect once on all data
    4. Per year: filter, finalize, and write to parquet
    """
    years = list(range(start_date.year, end_date.year + 1))

    # Step 1: Lazy scan all sources
    returns_lazy = database.barra_returns_table.read()
    specific_returns_lazy = database.barra_specific_returns_table.read()
    risk_lazy = database.barra_risk_table.read()
    volume_lazy = database.barra_volume_table.read()
    asset_ids_lazy = database.asset_ids_table.read_id_file()
    barra_ids_lazy = database.barra_ids_table.read_id_file()
    
    # Step 2: Chain all lazy joins on all data
    combined = (
        returns_lazy
        .join(
            specific_returns_lazy,
            on=["date", "barrid"],
            how="left"
        )
        .join(
            risk_lazy,
            on=["date", "barrid"],
            how="left"
        )
        .join(
            volume_lazy,
            on=["date", "barrid"],
            how="left"
        )
    )

    # Step 3: Join_asof with asset_ids
    combined = combined.join_asof(
        asset_ids_lazy.select(["barrid", "rootid", "issuerid", "instrument", "name", "iso_country_code", "iso_currency_code", "start_date", "end_date"]),
        left_on="date",
        right_on="start_date",
        by="barrid",
        strategy="backward"
    )

    # Step 4: Join_asof with barra_ids (pivoted)
    barra_ids_pivoted = (
        barra_ids_lazy
        .with_columns(
            pl.when(pl.col("asset_id_type") == "LOCAL_ID")
            .then(pl.lit("ticker"))
            .otherwise(pl.col("asset_id_type").str.to_lowercase())
            .alias("asset_id_type")
        )
        .pivot(
            on="asset_id_type",
            on_columns=["ticker", "cusip", "isin", "cisn"],
            values="asset_id",
            aggregate_function="first"
        )
        .select(["barrid", "cusip", "isin", "cisn", "ticker", "start_date", "end_date"])
    )

    combined = combined.join_asof(
        barra_ids_pivoted,
        left_on="date",
        right_on="start_date",
        by="barrid",
        strategy="backward"
    )

    # Step 5: FTSE Russell merge with rebalance-aware forward fill
    ftse_russell_lazy = database.ftse_russell_table.read()

    # Join FTSE on cusip
    combined = combined.join(ftse_russell_lazy, on=["date", "cusip"], how="left")

    # Identify rebalance dates (dates where any asset has Russell membership)
    russell_rebalance_dates = (
        combined
        .filter(pl.col("russell_1000") | pl.col("russell_2000"))
        .select("date", pl.lit(True).alias("russell_rebalance"))
        .unique()
    )

    # On rebalance dates, fill null memberships with False; then forward fill; compute in_universe
    combined = (
        combined
        .join(russell_rebalance_dates, on="date", how="left")
        .with_columns(
            pl.when(pl.col("russell_rebalance")).then(
                pl.col("russell_1000", "russell_2000").fill_null(False)
            )
        )
        .sort(["barrid", "date"])
        .with_columns(
            pl.col("russell_1000", "russell_2000")
            .fill_null(strategy="forward")
            .over("barrid")
        )
        .with_columns(
            pl.col("russell_1000").or_(pl.col("russell_2000")).alias("in_universe")
        )
        .drop("russell_rebalance")
    )

    # Step 6: Collect once on all data
    combined_eager = combined.collect()

    # Step 7: Per year, filter and write to parquet
    for year in tqdm(years, desc="Assets Backfill"):
        database.assets_table.delete(year)
        year_data = combined_eager.filter(pl.col("date").dt.year().eq(year))
        year_data.write_parquet(database.assets_table._file_path(year))

if __name__ == '__main__':
    from pipelines.utils.enums import DatabaseName
    start = date(1995, 1, 1)
    end = date(2025, 12, 31)
    db = Database(DatabaseName.DEVELOPMENT)
    assets_backfill_flow(start, end, db)
