from datetime import date
import os

import polars as pl
import wrds
from tqdm import tqdm

from pipelines.utils import russell_columns
from pipelines.utils.tables import Database


COMPUSTAT_CIK_SCHEMA = {
    "date": pl.Date,
    "cusip": pl.String,
    "russell2000": pl.String,
    "russell1000": pl.String,
    "cik": pl.String,
}


def load_compustat_cik_df(start_date: date, end_date: date, user: str) -> pl.DataFrame:
    """Load FTSE Russell membership plus Compustat-backed CIKs."""
    wrds_db = wrds.Connection(wrds_username=user)

    df = wrds_db.raw_sql(
        f"""
            SELECT
                a.date,
                a.cusip,
                a.russell2000,
                a.russell1000,
                b.cik
            FROM ftse_russell_us.idx_holdings_us a
            LEFT JOIN (
                SELECT
                    names.cusip AS cusip,
                    LPAD(CAST(company.cik AS varchar), 10, '0') AS cik
                FROM comp.names AS names
                INNER JOIN comp.company AS company
                    ON names.gvkey = company.gvkey
                WHERE names.cusip IS NOT NULL
                    AND company.cik IS NOT NULL
            ) b
                ON LEFT(a.cusip, 8) = LEFT(b.cusip, 8)
            WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY a.cusip, a.date
        """
    )
    return pl.from_pandas(df, schema_overrides=COMPUSTAT_CIK_SCHEMA)


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """Clean and standardize Compustat CIK dataframe."""
    return df.rename(russell_columns, strict=False).with_columns(
        pl.col("russell_2000", "russell_1000").eq("Y"),
        pl.col("cik").cast(pl.String).str.strip_chars(),
    )


def compustat_cik_backfill_flow(
    start_date: date, end_date: date, database: Database
) -> None:
    """
    Materialize the dated FTSE Russell + CIK table used by the 10-K flow.
    """
    user = os.getenv("WRDS_USER")
    raw_df = load_compustat_cik_df(start_date=start_date, end_date=end_date, user=user)
    clean_df = clean(raw_df)

    years = list(range(start_date.year, end_date.year + 1))

    for year in tqdm(years, desc="Compustat CIK"):
        year_data = clean_df.filter(pl.col("date").dt.year() == year)

        database.compustat_cik_table.create_if_not_exists(year)
        database.compustat_cik_table.upsert(year, year_data)
