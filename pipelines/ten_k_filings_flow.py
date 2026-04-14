from datetime import date
import os
from edgar import set_identity
import polars as pl

from pipelines.utils.ten_k_utils import (
    TEN_K_RECHECK_TRADING_DAYS,
    fetch_filings_for_rows,
    load_all_existing_ten_k_filings_df,
    load_existing_ten_k_filings_df,
    load_ftse_cik_df,
    load_today_ftse_cik_df,
    normalize_all_existing_ten_k_files,
    normalize_cik,
    year_bounds,
    trading_day_cutoff,
)
from pipelines.utils.tables import Database

def _daily_update_candidate_df(database: Database, current_date: date) -> pl.DataFrame:
    cutoff_date = trading_day_cutoff(current_date, TEN_K_RECHECK_TRADING_DAYS)

    cik_df = (
        load_today_ftse_cik_df(database, current_date)
        .with_columns(pl.col("cik").map_elements(normalize_cik, return_dtype=pl.String))
        .filter(pl.col("cik").is_not_null())
        .sort(["cusip", "date", "cik"])
        # Daily refresh mode should reason about the current Russell universe at
        # the security/CUSIP level, not every possible CIK variant that might
        # exist for the same issuer in Compustat.
        .unique(subset=["cusip"], keep="last")
        .select("cusip", "cik")
        .sort(["cusip", "cik"])
    )

    if cik_df.is_empty():
        return cik_df

    latest_existing_df = (
        load_all_existing_ten_k_filings_df(database)
        .filter(pl.col("filing_date").is_not_null())
        .sort(["cusip", "filing_date"])
        .unique(subset=["cusip"], keep="last")
        .select(["cusip", "filing_date"])
        .rename({"filing_date": "last_filing_date"})
    )

    # Only re-query companies whose most recent saved 10-K is at least
    # 245 trading days old, or that have never been stored at all.
    return (
        cik_df
        .join(latest_existing_df, on="cusip", how="left")
        .filter(
            pl.col("last_filing_date").is_null()
            | pl.col("last_filing_date").le(cutoff_date)
        )
    )


def ten_k_filings_daily_flow(current_date: date, database: Database) -> None:
    candidate_df = _daily_update_candidate_df(database, current_date)

    if candidate_df.is_empty():
        return

    lookback_start = date(current_date.year - 2, 1, 1)
    rows = candidate_df.select("cusip", "cik").to_dicts()
    results = fetch_filings_for_rows(
        rows,
        start_date=lookback_start,
        end_date=current_date,
        year=None,
        desc=f"10-K Filings Daily Update {current_date}",
    )

    filings_df = (
        pl.from_dicts(results, infer_schema_length=10000)
        .filter(pl.col("filing_date").is_not_null())
        .sort(["year", "cusip", "cik"])
    )

    if filings_df.is_empty():
        return

    for year_df in filings_df.partition_by("year", maintain_order=True):
        year = int(year_df["year"][0])
        database.ten_k_filings_table.create_if_not_exists(year)
        database.ten_k_filings_table.upsert(year, year_df.sort(["cusip", "cik"]))


def ten_k_filings_flow(
    start_date: date, end_date: date, database: Database, today_mode: bool = False
) -> None:
    identity = os.getenv("SEC_IDENTITY")
    if identity is None:
        raise EnvironmentError(
            "Missing required environment variable: SEC_IDENTITY. "
            "Check your .env file."
        )

    set_identity(identity)
    normalize_all_existing_ten_k_files(database)

    if today_mode:
        ten_k_filings_daily_flow(end_date, database)
        return

    years = list(range(start_date.year, end_date.year + 1))

    for year in years:
        year_start, year_end = year_bounds(year, start_date, end_date)

        cik_df = (
            load_ftse_cik_df(database, year_start, year_end)
            .with_columns(pl.col("cik").map_elements(normalize_cik, return_dtype=pl.String))
            .filter(pl.col("cik").is_not_null())
            .sort(["cusip", "date"])
            # Deduplicate to the latest Russell membership observation for each
            # CUSIP/CIK pair in the year before querying EDGAR.
            .unique(subset=["cusip", "cik"], keep="last")
            .select("cusip", "cik")
            .sort(["cusip", "cik"])
        )

        if cik_df.is_empty():
            continue

        existing_df = (
            load_existing_ten_k_filings_df(database, year)
            .filter(pl.col("filing_date").is_not_null())
            .select("cusip", "cik")
            .unique()
        )

        # Re-runs only query names that do not already have a filing saved for
        # the target year, which keeps daily/yearly refreshes lightweight.
        cik_df = cik_df.join(existing_df, on=["cusip", "cik"], how="anti")

        if cik_df.is_empty():
            continue

        rows = cik_df.to_dicts()
        results = fetch_filings_for_rows(
            rows,
            start_date=year_start,
            end_date=year_end,
            year=year,
            desc=f"10-K Filings {year}",
        )

        filings_df = pl.from_dicts(results, infer_schema_length=10000).sort(["cusip", "cik"])

        database.ten_k_filings_table.create_if_not_exists(year)
        database.ten_k_filings_table.upsert(year, filings_df)
