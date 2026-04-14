from datetime import date

import polars as pl

from pipelines.utils.tables import Database, Table


def load_signal_assets_df(
    database: Database, start_date: date | None = None, end_date: date | None = None
) -> pl.DataFrame:

    assets_lazy = database.assets_table.read()

    filter_exprs = [pl.col("in_universe") == True]
    if start_date is not None and end_date is not None:
        filter_exprs.append(pl.col("date").is_between(start_date, end_date))

    needed_cols = (
        assets_lazy
        .select([
            "date",
            "barrid",
            "cusip",
            "ticker",
            "price",
            "return",
            "specific_return",
            "specific_risk",
            "predicted_beta",
            "daily_volume",
            "in_universe",
        ])
        .filter(*filter_exprs)
        .with_columns(
            pl.col("return").truediv(100),
            pl.col("specific_return").truediv(100),
            pl.col("specific_risk").truediv(100),
        )
    )

    return needed_cols.collect().sort(["barrid", "date"])


def _load_existing_table_df(table: Table) -> pl.DataFrame:
    try:
        return table.read_id_file().collect()
    except Exception:
        return pl.DataFrame(schema=table._schema)


def _merge_signal_subset(
    table: Table, new_df: pl.DataFrame, signal_names: list[str]
) -> pl.DataFrame:
    
    existing_df = _load_existing_table_df(table)

    if existing_df.is_empty():
        return new_df

    # Keep the rows for all signal families we are not updating in this run.
    keep_existing_df = existing_df.filter(~pl.col("signal_name").is_in(signal_names))

    if new_df.is_empty():
        return keep_existing_df

    # Append the refreshed rows for the target signal subset.
    return pl.concat([keep_existing_df, new_df], how="vertical_relaxed")


def write_signal_subset_outputs(
    database: Database,
    signal_names: list[str],
    signals_df: pl.DataFrame,
    scores_df: pl.DataFrame,
    alphas_df: pl.DataFrame,
) -> None:
    
    merged_signals_df = _merge_signal_subset(database.signals_table, signals_df, signal_names)
    merged_scores_df = _merge_signal_subset(database.scores_table, scores_df, signal_names)
    merged_alphas_df = _merge_signal_subset(database.alpha_table, alphas_df, signal_names)

    database.signals_table.overwrite(merged_signals_df)
    database.scores_table.overwrite(merged_scores_df)
    database.alpha_table.overwrite(merged_alphas_df)
