from datetime import date
import polars as pl
from pipelines.utils.tables import Database
from pipelines.signals import SIGNALS


def signals_flow(start_date: date, end_date: date, database: Database) -> None:
    """
    Compute signals, scores, and alphas from assets table.

    Since signals have lookback windows (up to 252 days), the entire history
    must be recomputed on every run.

    Strategy:
    1. Lazy scan assets_table (all years needed for rolling window lookback)
    2. Filter to in_universe=True and select needed columns
    3. Collect eagerly (rolling windows require full sorted history)
    4. For each signal: compute signal_value, score (z-score), and alpha
    5. Write single parquet file per table (all years)
    """

    # Step 1: Lazy scan assets_table
    assets_lazy = database.assets_table.read()

    # Step 2: Filter and select needed columns
    needed_cols = assets_lazy.select([
        "date",
        "barrid",
        "return",
        "predicted_beta",
        "specific_risk",
        "in_universe"
    ]).filter(pl.col("in_universe") == True)

    # Step 3: Collect eagerly (rolling windows need full history in memory)
    assets_df = needed_cols.collect().sort(["barrid", "date"])

    # Lists to accumulate results for each table
    signals_rows = []
    scores_rows = []
    alphas_rows = []

    # Step 4: Compute each signal
    for signal_name, signal_config in SIGNALS.items():
        # Add signal column
        df_with_signal = assets_df.with_columns(signal_config["expr"])

        # Extract signal value (the expr has .alias(signal_name))
        signal_data = df_with_signal.select([
            "date",
            "barrid",
            pl.col(signal_name).alias("signal_value"),
            "specific_risk"
        ]).drop_nulls("signal_value")

        # Build signals_table rows
        signals_row = signal_data.select([
            "date",
            "barrid",
            pl.lit(signal_name).alias("signal_name"),
            "signal_value"
        ])
        signals_rows.append(signals_row)

        # Compute score using signal-specific scorer
        scorer = signal_config["scorer"]
        score_data = scorer(signal_data)

        # Build scores_table rows
        scores_row = score_data.select([
            "date",
            "barrid",
            pl.lit(signal_name).alias("signal_name"),
            "score"
        ])
        scores_rows.append(scores_row)

        # Compute alpha using signal-specific alphatizer
        alphatizer = signal_config["alphatizer"]
        alpha_data = alphatizer(score_data)

        # Build alphas_table rows
        alphas_row = alpha_data.select([
            "date",
            "barrid",
            pl.lit(signal_name).alias("signal_name"),
            "alpha"
        ])
        alphas_rows.append(alphas_row)

    # Concatenate all rows for each table
    signals_df = pl.concat(signals_rows)
    scores_df = pl.concat(scores_rows)
    alphas_df = pl.concat(alphas_rows)

    # Step 5: Write single file per table (all years)
    database.signals_table.overwrite(signals_df)
    database.scores_table.overwrite(scores_df)
    database.alpha_table.overwrite(alphas_df)


if __name__ == "__main__":
    from pipelines.utils.enums import DatabaseName
    start = date(1995, 1, 1)
    end = date(2025, 12, 31)
    db = Database(DatabaseName.DEVELOPMENT)
    signals_flow(start, end, db)
