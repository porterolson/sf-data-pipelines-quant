from datetime import date
import polars as pl
from tqdm import tqdm
from pipelines.utils.tables import Database
from pipelines.signals import SIGNALS, build_ten_k_similarity_df

TEN_K_HOLDING_DAYS = 245


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

    # Lazy scan assets_table
    assets_lazy = database.assets_table.read()

    # Filter and select needed columns
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
            "in_universe"
        ])
        .filter(
            pl.col("in_universe") == True
        )
        .with_columns(
            pl.col("return").truediv(100),
            pl.col("specific_return").truediv(100),
            pl.col("specific_risk").truediv(100),
        )
    )

    # Collect eagerly (rolling windows need full history in memory)
    assets_df = needed_cols.collect().sort(["barrid", "date"])


    try:
        ten_k_df = (
            database.ten_k_filings_table.read()
            .select(["year", "cusip", "cik", "filing_date", "item_1a"])
            .collect()
            .unique(subset=["cusip", "cik", "year", "filing_date"], keep="last")
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load ten_k_filings parquet files. "
            "This is often caused by schema mismatches across yearly files."
        ) from exc

    ten_k_similarity_df = build_ten_k_similarity_df(ten_k_df)

    if ten_k_similarity_df.is_empty():
        assets_df = assets_df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("ten_k_similarity_value"),
            pl.lit(None, dtype=pl.Date).alias("ten_k_similarity_filing_date"),
        )
    else:
        assets_df = (
            assets_df
            .with_columns(pl.col("cusip").str.slice(0, 8).alias("cusip"))
            .sort(["cusip", "date"])
            .join_asof(
                ten_k_similarity_df
                .rename({"filing_date": "ten_k_similarity_filing_date"})
                .sort(["cusip", "ten_k_similarity_filing_date"]),
                left_on="date",
                right_on="ten_k_similarity_filing_date",
                by="cusip",
                strategy="backward",
            )
            .with_columns(
                # A filed 10-K should stay tradable for a fixed holding window,
                # but the score itself is still computed only on the event date.
                pl.when(
                    pl.col("ten_k_similarity_filing_date").is_not_null()
                    & (
                        (pl.col("date") - pl.col("ten_k_similarity_filing_date"))
                        <= pl.duration(days=TEN_K_HOLDING_DAYS)
                    )
                )
                .then(pl.col("ten_k_similarity_value"))
                .otherwise(None)
                .alias("ten_k_similarity_value")
            )
            .sort(["barrid", "date"])
        )

    ten_k_panel_df = (
        assets_df
        .with_columns(SIGNALS["ten_k_similarity"]["expr"])
        .filter(pl.col("ten_k_similarity").is_not_null())
        .with_columns(pl.col("ten_k_similarity").alias("signal_value"))
        .unique(subset=["date", "barrid"], keep="last")
        .sort(["barrid", "date"])
    )

    ten_k_event_signal_df = (
        ten_k_panel_df
        # Legacy parity: score the KL signal only across names that actually
        # filed on that date, not across the whole 245-day active panel.
        .filter(pl.col("date").eq(pl.col("ten_k_similarity_filing_date")))
        .filter(
            pl.col("ten_k_similarity").is_not_null(),
            pl.col("specific_risk").is_not_null(),
        )
        .unique(subset=["date", "barrid"], keep="last")
        .sort(["barrid", "date"])
    )

    ten_k_score_df = (
        SIGNALS["ten_k_similarity"]["scorer"](ten_k_event_signal_df)
        if not ten_k_event_signal_df.is_empty()
        else pl.DataFrame(
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                "cusip": pl.String,
                "ten_k_similarity": pl.Float64,
                "signal_value": pl.Float64,
                "score": pl.Float64,
                "specific_risk": pl.Float64,
                "ten_k_similarity_filing_date": pl.Date,
            }
        )
    )

    ten_k_event_df = (
        ten_k_score_df
        .filter(
            pl.col("ten_k_similarity").is_not_null(),
            pl.col("specific_risk").is_not_null(),
            pl.col("score").is_not_null(),
            # A NaN score usually means the event-date cross section was too
            # thin to form a meaningful z-score, so treat it as no event alpha.
            pl.col("score").is_nan().not_(),
        )
        .unique(subset=["date", "barrid"], keep="last")
        .sort(["barrid", "date"])
    )

    if ten_k_event_df.is_empty():
        ten_k_alpha_df = pl.DataFrame(
            schema={
                "barrid": pl.String,
                "date": pl.Date,
                "alpha": pl.Float64,
            }
        )
    else:
        ten_k_alpha_df = (
            SIGNALS["ten_k_similarity"]["alphatizer"](ten_k_event_df)
            .select("barrid", "date", "alpha")
            .filter(
                pl.col("alpha").is_not_null(),
                pl.col("alpha").is_nan().not_(),
            )
            .unique(subset=["date", "barrid"], keep="last")
            .sort(["barrid", "date"])
        )

        ten_k_alpha_df = (
            assets_df
            .select("date", "barrid")
            .unique(subset=["date", "barrid"], keep="last")
            .join(ten_k_alpha_df, on=["date", "barrid"], how="left")
            .sort(["barrid", "date"])
            .with_columns(
                # Shift by one trading day and then simply forward-fill that same
                # event alpha for 245 trading days; we do not recompute the alpha
                # daily once the filing-date signal has been formed.
                pl.col("alpha")
                .shift(1)
                .forward_fill(limit=TEN_K_HOLDING_DAYS)
                .over("barrid")
                .alias("alpha")
            )
            .with_columns(pl.col("alpha").fill_null(0.0))
            .unique(subset=["date", "barrid"], keep="last")
        )

    # Lists to accumulate results for each table
    signals_rows = []
    scores_rows = []
    alphas_rows = []

    # Compute each signal
    for signal_name, signal_config in tqdm(SIGNALS.items(), desc="Signals", total=len(SIGNALS)):
        if signal_name == "ten_k_similarity":
            signal_df = (
                ten_k_panel_df
                .select([
                    "date",
                    "barrid",
                    "signal_value",
                ])
                .unique(subset=["date", "barrid"], keep="last")
            )
            signals_rows.append(
                signal_df.select([
                    "date",
                    "barrid",
                    pl.lit(signal_name).alias("signal_name"),
                    "signal_value"
                ]).unique(subset=["date", "barrid", "signal_name"], keep="last")
            )

            if not ten_k_score_df.is_empty():
                scores_rows.append(
                    ten_k_score_df
                    .select([
                        "date",
                        "barrid",
                        pl.lit(signal_name).alias("signal_name"),
                        "score"
                    ])
                    .unique(subset=["date", "barrid", "signal_name"], keep="last")
                )

            if not ten_k_alpha_df.is_empty():
                alphas_rows.append(
                    ten_k_alpha_df
                    .select([
                        "date",
                        "barrid",
                        pl.lit(signal_name).alias("signal_name"),
                        "alpha"
                    ])
                    .unique(subset=["date", "barrid", "signal_name"], keep="last")
                )
            continue

        signal_df = (
            assets_df
            .with_columns(signal_config["expr"])
            .with_columns(pl.col(signal_name).alias("signal_value"))
        ).filter(
                pl.col(signal_name).is_not_null(),
                pl.col("predicted_beta").is_not_null(),
                pl.col("specific_risk").is_not_null(),
            )
        signals_rows.append(signal_df.select([
            "date",
            "barrid",
            pl.lit(signal_name).alias("signal_name"),
            "signal_value"
        ]).unique(subset=["date", "barrid", "signal_name"], keep="last"))
        
        # Compute score using signal-specific scorer
        scorer = signal_config["scorer"]
        score_df = scorer(signal_df)
        scores_rows.append(score_df.select([
            "date",
            "barrid",
            pl.lit(signal_name).alias("signal_name"),
            "score"
        ]).unique(subset=["date", "barrid", "signal_name"], keep="last"))

        # Compute alpha using signal-specific alphatizer
        alphatizer = signal_config["alphatizer"]
        alpha_df = alphatizer(score_df)
        alphas_rows.append(alpha_df.select([
            "date",
            "barrid",
            pl.lit(signal_name).alias("signal_name"),
            "alpha"
        ]).unique(subset=["date", "barrid", "signal_name"], keep="last"))

    # Concatenate all rows for each table
    signals_df = pl.concat(signals_rows)
    scores_df = pl.concat(scores_rows)
    alphas_df = pl.concat(alphas_rows)

    if ten_k_panel_df.height > 0:
        final_signal_names = set(signals_df["signal_name"].unique().to_list())
        if "ten_k_similarity" not in final_signal_names:
            raise RuntimeError(
                "ten_k_similarity rows were built upstream but are missing from final signals_df."
            )

    # Write single file per table (all years)
    database.signals_table.overwrite(signals_df)
    database.scores_table.overwrite(scores_df)
    database.alpha_table.overwrite(alphas_df)


if __name__ == "__main__":
    from pipelines.utils.enums import DatabaseName
    start = date(1995, 1, 1)
    end = date(2025, 12, 31)
    db = Database(DatabaseName.DEVELOPMENT)
    signals_flow(start, end, db)
