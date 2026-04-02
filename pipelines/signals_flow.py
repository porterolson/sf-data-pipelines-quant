from datetime import date
import polars as pl
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
    assets_df = (
        needed_cols
        .collect()
        .unique(subset=["date", "barrid"], keep="last")
        .sort(["barrid", "date"])
    )

    try:
        ten_k_df = (
            database.ten_k_filings_table.read()
            .select(["year", "cusip", "cik", "filing_date", "item_1a"])
            .collect()
            .unique(subset=["cusip", "cik", "year", "filing_date"], keep="last")
        )
        print(f"10-K input rows: {ten_k_df.height}")
        print(
            "10-K rows with non-null Item 1A: "
            f"{ten_k_df.filter(pl.col('item_1a').is_not_null()).height}"
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load ten_k_filings parquet files. "
            "This is often caused by schema mismatches across yearly files."
        ) from exc

    ten_k_similarity_df = build_ten_k_similarity_df(ten_k_df)
    print(f"10-K similarity rows: {ten_k_similarity_df.height}")
    print(
        "10-K similarity non-null rows: "
        f"{ten_k_similarity_df.filter(pl.col('ten_k_similarity_value').is_not_null()).height}"
    )

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

    ten_k_signal_df = (
        assets_df
        .with_columns(SIGNALS["ten_k_similarity"]["expr"])
        .filter(pl.col("ten_k_similarity").is_not_null())
        .with_columns(pl.col("ten_k_similarity").alias("signal_value"))
        .unique(subset=["date", "barrid"], keep="last")
        .sort(["barrid", "date"])
    )
    print(f"10-K signal rows after join/filter: {ten_k_signal_df.height}")

    ten_k_score_df = (
        SIGNALS["ten_k_similarity"]["scorer"](ten_k_signal_df)
        if not ten_k_signal_df.is_empty()
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
    if ten_k_score_df.is_empty():
        print("10-K score rows: 0")
    else:
        print(f"10-K score rows: {ten_k_score_df.height}")
        print(
            "10-K non-null score rows: "
            f"{ten_k_score_df.filter(pl.col('score').is_not_null()).height}"
        )

    ten_k_event_df = (
        ten_k_score_df
        .filter(pl.col("date").eq(pl.col("ten_k_similarity_filing_date")))
        .filter(
            pl.col("ten_k_similarity").is_not_null(),
            pl.col("specific_risk").is_not_null(),
            pl.col("score").is_not_null(),
        )
        .unique(subset=["date", "barrid"], keep="last")
        .sort(["barrid", "date"])
    )
    print(f"10-K event rows: {ten_k_event_df.height}")

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
                pl.col("alpha")
                .shift(1)
                .forward_fill(limit=TEN_K_HOLDING_DAYS)
                .over("barrid")
                .alias("alpha")
            )
            .with_columns(pl.col("alpha").fill_null(0.0))
            .unique(subset=["date", "barrid"], keep="last")
        )
    print(f"10-K daily alpha rows: {ten_k_alpha_df.height}")
    print("10-K branch completed")

    # Lists to accumulate results for each table
    signals_rows = []
    scores_rows = []
    alphas_rows = []

    # Compute each signal
    for signal_name, signal_config in SIGNALS.items():
        print(f"Processing signal: {signal_name}")
        if signal_name == "ten_k_similarity":
            signal_df = (
                ten_k_signal_df
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
            print("Finished signal: ten_k_similarity")
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
        print(f"Finished signal: {signal_name}")

    # Concatenate all rows for each table
    print("Starting final concatenation")
    signals_df = pl.concat(signals_rows)
    scores_df = pl.concat(scores_rows)
    alphas_df = pl.concat(alphas_rows)
    print("Finished final concatenation")

    print("Final signals counts:", signals_df.group_by("signal_name").len().sort("signal_name").to_dicts())
    print("Final scores counts:", scores_df.group_by("signal_name").len().sort("signal_name").to_dicts())
    print("Final alphas counts:", alphas_df.group_by("signal_name").len().sort("signal_name").to_dicts())

    if ten_k_signal_df.height > 0:
        final_signal_names = set(signals_df["signal_name"].unique().to_list())
        if "ten_k_similarity" not in final_signal_names:
            raise RuntimeError(
                "ten_k_similarity rows were built upstream but are missing from final signals_df."
            )

    # Write single file per table (all years)
    print("Starting overwrite")
    database.signals_table.overwrite(signals_df)
    database.scores_table.overwrite(scores_df)
    database.alpha_table.overwrite(alphas_df)
    print("Finished overwrite")


if __name__ == "__main__":
    from pipelines.utils.enums import DatabaseName
    start = date(1995, 1, 1)
    end = date(2025, 12, 31)
    db = Database(DatabaseName.DEVELOPMENT)
    signals_flow(start, end, db)
