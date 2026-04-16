from datetime import date
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pipelines.signals import ic_alphatizer, zscore_scorer
from pipelines.signal_output_utils import load_signal_assets_df, write_signal_subset_outputs
from pipelines.utils.tables import Database

TEN_K_SIGNAL_NAME = "ten_k_similarity"
TEN_K_HOLDING_DAYS = 245


def build_ten_k_similarity_df(ten_k_df: pl.DataFrame) -> pl.DataFrame:
    if ten_k_df.is_empty():
        return pl.DataFrame(
            schema={
                "cusip": pl.String,
                "filing_date": pl.Date,
                "ten_k_similarity_value": pl.Float64,
            }
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        stop_words="english",
    )

    ten_k_df = (
        ten_k_df
        .filter(
            pl.col("cusip").is_not_null(),
            pl.col("filing_date").is_not_null(),
        )
        .with_columns(pl.col("cusip").str.slice(0, 8).alias("cusip"))
        .sort(
            ["cik", "year", "filing_date", "cusip"],
            descending=[False, True, True, False],
        )
        .unique(subset=["cik", "year"], keep="last")
        .sort(["cik", "year"], descending=[False, True])
    )

    rows: list[dict[str, object]] = []
    cik_list = ten_k_df["cik"].drop_nulls().unique().sort().to_list()

    for cik in cik_list:
        sub = ten_k_df.filter(pl.col("cik").eq(cik))
        year_item = sub.select(["cusip", "filing_date", "item_1a", "year"])
        year_list = year_item["year"].to_list()

        for year in year_list:
            current_row = year_item.filter(pl.col("year").eq(year))
            prior_row = year_item.filter(pl.col("year").eq(year - 1))

            cusip = current_row.select("cusip").item()
            filing_date = current_row.select("filing_date").item()

            try:
                doc_1 = current_row.select("item_1a").item()
                doc_2 = prior_row.select("item_1a").item()
            except ValueError:
                rows.append(
                    {
                        "cusip": cusip,
                        "filing_date": filing_date,
                        "ten_k_similarity_value": None,
                    }
                )
                continue

            try:
                tfidf_matrix = vectorizer.fit_transform([doc_1, doc_2])
                similarity = float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])
            except AttributeError:
                similarity = None

            rows.append(
                {
                    "cusip": cusip,
                    "filing_date": filing_date,
                    "ten_k_similarity_value": similarity,
                }
            )

    return pl.from_dicts(
        rows,
        schema={
            "cusip": pl.String,
            "filing_date": pl.Date,
            "ten_k_similarity_value": pl.Float64,
        },
    ).sort(["cusip", "filing_date"])


def build_ten_k_signal_outputs(
    assets_df: pl.DataFrame, database: Database
) -> tuple[pl.DataFrame | None, pl.DataFrame | None, pl.DataFrame | None]:
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
        assets_with_ten_k_df = assets_df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("ten_k_similarity_value"),
            pl.lit(None, dtype=pl.Date).alias("ten_k_similarity_filing_date"),
        )
    else:
        assets_with_ten_k_df = (
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

    ten_k_panel_df = (
        assets_with_ten_k_df
        .with_columns(pl.col("ten_k_similarity_value").alias(TEN_K_SIGNAL_NAME))
        .filter(pl.col(TEN_K_SIGNAL_NAME).is_not_null())
        .with_columns(pl.col(TEN_K_SIGNAL_NAME).alias("signal_value"))
        .unique(subset=["date", "barrid"], keep="last")
        .sort(["barrid", "date"])
    )

    if ten_k_panel_df.is_empty():
        return None, None, None

    ten_k_event_signal_df = (
        ten_k_panel_df
        .filter(pl.col("date").eq(pl.col("ten_k_similarity_filing_date")))
        .filter(
            pl.col(TEN_K_SIGNAL_NAME).is_not_null(),
            pl.col("specific_risk").is_not_null(),
        )
        .unique(subset=["date", "barrid"], keep="last")
        .sort(["barrid", "date"])
    )

    ten_k_score_df = (
        zscore_scorer(ten_k_event_signal_df)
        if not ten_k_event_signal_df.is_empty()
        else pl.DataFrame(
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                "cusip": pl.String,
                TEN_K_SIGNAL_NAME: pl.Float64,
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
            pl.col(TEN_K_SIGNAL_NAME).is_not_null(),
            pl.col("specific_risk").is_not_null(),
            pl.col("score").is_not_null(),
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
            ic_alphatizer(ten_k_event_df)
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
                .forward_fill(limit=TEN_K_HOLDING_DAYS)
                .over("barrid")
                .alias("alpha")
            )
            .with_columns(pl.col("alpha").fill_null(0.0))
            .unique(subset=["date", "barrid"], keep="last")
        )

    signal_rows = (
        ten_k_panel_df
        .select([
            "date",
            "barrid",
            pl.lit(TEN_K_SIGNAL_NAME).alias("signal_name"),
            "signal_value",
        ])
        .unique(subset=["date", "barrid", "signal_name"], keep="last")
    )

    score_rows = None
    if not ten_k_score_df.is_empty():
        score_rows = (
            ten_k_score_df
            .select([
                "date",
                "barrid",
                pl.lit(TEN_K_SIGNAL_NAME).alias("signal_name"),
                "score",
            ])
            .unique(subset=["date", "barrid", "signal_name"], keep="last")
        )

    alpha_rows = None
    if not ten_k_alpha_df.is_empty():
        alpha_rows = (
            ten_k_alpha_df
            .select([
                "date",
                "barrid",
                pl.lit(TEN_K_SIGNAL_NAME).alias("signal_name"),
                "alpha",
            ])
            .unique(subset=["date", "barrid", "signal_name"], keep="last")
        )

    return signal_rows, score_rows, alpha_rows


def ten_k_signals_flow(start_date: date, end_date: date, database: Database) -> None:
    assets_df = load_signal_assets_df(database, start_date, end_date)
    signal_rows, score_rows, alpha_rows = build_ten_k_signal_outputs(assets_df, database)

    signals_df = (
        signal_rows
        if signal_rows is not None
        else pl.DataFrame(schema=database.signals_table._schema)
    )
    scores_df = (
        score_rows
        if score_rows is not None
        else pl.DataFrame(schema=database.scores_table._schema)
    )
    alphas_df = (
        alpha_rows
        if alpha_rows is not None
        else pl.DataFrame(schema=database.alpha_table._schema)
    )

    write_signal_subset_outputs(
        database,
        signal_names=[TEN_K_SIGNAL_NAME],
        signals_df=signals_df,
        scores_df=scores_df,
        alphas_df=alphas_df,
    )


if __name__ == "__main__":
    from pipelines.utils.enums import DatabaseName

    start = date(1995, 1, 1)
    end = date(2025, 12, 31)
    db = Database(DatabaseName.DEVELOPMENT)
    ten_k_signals_flow(start, end, db)
