from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from glob import glob
import os
import time

from edgar import Company
import polars as pl
from tqdm import tqdm

from pipelines.utils import get_last_market_date
from pipelines.utils.tables import Database

MAX_WORKERS = 4
SLEEP_BETWEEN = 0.2
TEN_K_RECHECK_TRADING_DAYS = 245
TEN_K_FILE_COLUMNS = [
    "year",
    "cusip",
    "cik",
    "form",
    "filing_date",
    "acceptance_datetime",
    "accession_number",
    "filing_url",
    "item_1a",
]


def normalize_cik(cik: str | None) -> str | None:
    if cik is None:
        return None

    cik = str(cik).strip()
    return cik.zfill(10) if cik else None


def safe_attr(obj, *names: str):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value

    return None


def safe_item(obj, key: str) -> str | None:
    try:
        value = obj[key]
    except Exception:
        return None

    if value is None:
        return None

    value = str(value).strip()
    return value or None


def normalize_date(value) -> date | None:
    if value is None:
        return None

    if isinstance(value, date):
        return value

    value = str(value).strip()
    if not value:
        return None

    return date.fromisoformat(value[:10])


def normalize_datetime(value) -> str | None:
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    value = str(value).strip()
    return value or None


def latest_filing(company: Company, start_date: date, end_date: date):
    filings = company.get_filings(form="10-K")

    if filings is None:
        return None

    try:
        filings = filings.filter(date=f"{start_date}:{end_date}")
    except Exception:
        pass

    try:
        if len(filings) == 0:
            return None
    except TypeError:
        pass

    latest = None

    try:
        latest = filings[0]
    except Exception:
        try:
            latest = filings.latest()
        except Exception:
            return None

    if latest is None:
        return None

    try:
        if not hasattr(latest, "form") and len(latest) > 0:
            latest = latest[0]
    except Exception:
        pass

    return latest


def load_ftse_cik_df(
    database: Database, start_date: date, end_date: date
) -> pl.DataFrame:
    return (
        database.compustat_cik_table.read()
        .filter(
            pl.col("date").is_between(start_date, end_date),
            pl.col("russell_1000").fill_null(False) | pl.col("russell_2000").fill_null(False),
        )
        .collect()
        .with_columns(pl.col("cusip").str.slice(0, 8).alias("cusip"))
    )


def normalize_existing_ten_k_df(
    database: Database, df: pl.DataFrame, year: int | None = None
) -> pl.DataFrame:
    columns = set(df.columns)

    if "report_date" in columns:
        df = df.drop("report_date")

    missing_columns = [col for col in TEN_K_FILE_COLUMNS if col not in df.columns]
    if missing_columns:
        df = df.with_columns([pl.lit(None).alias(col) for col in missing_columns])

    df = (
        df
        .with_columns(pl.col("cusip").cast(pl.String).str.slice(0, 8).alias("cusip"))
        .select(TEN_K_FILE_COLUMNS)
    )

    if year is not None and "report_date" in columns:
        df.write_parquet(database.ten_k_filings_table._file_path(year))

    return df


def load_existing_ten_k_filings_df(database: Database, year: int) -> pl.DataFrame:
    try:
        df = database.ten_k_filings_table.read(year).collect()
        return normalize_existing_ten_k_df(database, df, year=year)
    except Exception:
        return pl.DataFrame(
            schema={
                "year": pl.Int64,
                "cusip": pl.String,
                "cik": pl.String,
                "form": pl.String,
                "filing_date": pl.Date,
                "acceptance_datetime": pl.String,
                "accession_number": pl.String,
                "filing_url": pl.String,
                "item_1a": pl.String,
            }
        )


def normalize_all_existing_ten_k_files(database: Database) -> None:
    table = database.ten_k_filings_table
    pattern = os.path.join(table._base_path, table._name, f"{table._name}_*.parquet")

    for file_path in glob(pattern):
        df = pl.read_parquet(file_path)
        if "report_date" not in df.columns:
            continue

        year = int(os.path.splitext(os.path.basename(file_path))[0].rsplit("_", 1)[1])
        normalized_df = normalize_existing_ten_k_df(database, df, year=None)
        normalized_df.write_parquet(table._file_path(year))


def load_all_existing_ten_k_filings_df(database: Database) -> pl.DataFrame:
    normalize_all_existing_ten_k_files(database)

    try:
        df = database.ten_k_filings_table.read().collect()
        return normalize_existing_ten_k_df(database, df, year=None)
    except Exception:
        return pl.DataFrame(
            schema={
                "year": pl.Int64,
                "cusip": pl.String,
                "cik": pl.String,
                "form": pl.String,
                "filing_date": pl.Date,
                "acceptance_datetime": pl.String,
                "accession_number": pl.String,
                "filing_url": pl.String,
                "item_1a": pl.String,
            }
        )


def trading_day_cutoff(current_date: date, lookback_days: int) -> date:
    previous_market_dates = [
        d for d in get_last_market_date(current_date=current_date, n_days=lookback_days) if d is not None
    ]

    if not previous_market_dates:
        return current_date

    return previous_market_dates[0]


def load_today_ftse_cik_df(database: Database, current_date: date) -> pl.DataFrame:
    latest_ftse_date = (
        database.compustat_cik_table.read()
        .filter(pl.col("date").le(current_date))
        .select(pl.col("date").max().alias("latest_date"))
        .collect()
        .item()
    )

    if latest_ftse_date is None:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "cusip": pl.String,
                "russell_2000": pl.Boolean,
                "russell_1000": pl.Boolean,
                "cik": pl.String,
            }
        )

    return (
        database.compustat_cik_table.read()
        .filter(
            pl.col("date").eq(latest_ftse_date),
            pl.col("russell_1000").fill_null(False) | pl.col("russell_2000").fill_null(False),
        )
        .collect()
        .with_columns(pl.col("cusip").str.slice(0, 8).alias("cusip"))
    )


def fetch_latest_ten_k_filing(
    row: dict[str, str | None], start_date: date, end_date: date, year: int | None = None
) -> dict[str, object]:
    result = {
        "year": year,
        "cusip": row["cusip"],
        "cik": row["cik"],
        "form": None,
        "filing_date": None,
        "acceptance_datetime": None,
        "accession_number": None,
        "filing_url": None,
        "item_1a": None,
    }

    cik = normalize_cik(row["cik"])
    if cik is None:
        return result

    try:
        time.sleep(SLEEP_BETWEEN)
        company = Company(cik)
        filing = latest_filing(company, start_date, end_date)

        if filing is None:
            return result

        filing_obj = None
        try:
            filing_obj = filing.obj()
        except Exception:
            filing_obj = None

        result.update(
            {
                "form": safe_attr(filing, "form"),
                "filing_date": normalize_date(safe_attr(filing, "filing_date")),
                "acceptance_datetime": normalize_datetime(
                    safe_attr(filing, "acceptance_datetime")
                ),
                "accession_number": safe_attr(
                    filing,
                    "accession_number",
                    "accession_no",
                ),
                "filing_url": safe_attr(
                    filing,
                    "filing_url",
                    "homepage_url",
                    "url",
                    "link",
                ),
                "item_1a": safe_item(filing_obj, "Item 1A") if filing_obj is not None else None,
            }
        )
        if result["year"] is None and result["filing_date"] is not None:
            result["year"] = result["filing_date"].year
    except Exception:
        return result

    return result


def year_bounds(year: int, start_date: date, end_date: date) -> tuple[date, date]:
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)

    return max(start_date, year_start), min(end_date, year_end)


def fetch_filings_for_rows(
    rows: list[dict[str, object]],
    start_date: date,
    end_date: date,
    year: int | None = None,
    desc: str | None = None,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(fetch_latest_ten_k_filing, row, start_date, end_date, year)
            for row in rows
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=desc,
        ):
            results.append(future.result())

    return results
