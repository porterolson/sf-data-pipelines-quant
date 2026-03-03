from datetime import date
import zipfile
import polars as pl
from io import BytesIO
from pipelines.utils import barra_schema, barra_columns
from pipelines.utils.barra_datasets import barra_returns
import os
from tqdm import tqdm
from pipelines.utils import get_last_market_date
from pipelines.utils.tables import Database


def load_barra_history_files(year: int) -> pl.DataFrame:
    zip_folder_path = barra_returns.history_zip_folder_path(year)
    file_name = barra_returns.file_name()

    with zipfile.ZipFile(zip_folder_path, "r") as zip_folder:
        dfs = [
            pl.read_csv(
                BytesIO(zip_folder.read(file)),
                skip_rows=1,
                separator="|",
                schema_overrides=barra_schema,
                try_parse_dates=True,
            )
            for file in zip_folder.namelist()
            if file.startswith(file_name)
        ]

    return pl.concat(dfs, how="vertical") if dfs else pl.DataFrame()


def load_current_barra_files() -> pl.DataFrame:
    dfs = []

    dates = get_last_market_date(n_days=60)

    for date_ in dates:
        zip_folder_path = barra_returns.daily_zip_folder_path(date_)
        file_name = barra_returns.file_name(date_)

        if os.path.exists(zip_folder_path):
            with zipfile.ZipFile(zip_folder_path, "r") as zip_folder:
                dfs.append(
                    pl.read_csv(
                        BytesIO(zip_folder.read(file_name)),
                        skip_rows=1,
                        separator="|",
                        schema_overrides=barra_schema,
                        try_parse_dates=True,
                    )
                )

    df = pl.concat(dfs)

    return df


def clean_barra_returns(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.rename(barra_columns, strict=False)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"))
        .filter(pl.col("barrid").ne("[End of File]"))
        .select(["barrid", "price", "market_cap", "price_source", "currency", "return", "date"])
        .sort(["barrid", "date"])
    )


def barra_returns_history_flow(
    start_date: date, end_date: date, database: Database
) -> None:
    years = list(range(start_date.year, end_date.year + 1))

    for year in tqdm(years, desc="Barra Returns"):
        raw_df = load_barra_history_files(year)
        clean_df = clean_barra_returns(raw_df)
        database.barra_returns_table.create_if_not_exists(year)
        database.barra_returns_table.upsert(year, clean_df)


def barra_returns_daily_flow(database: Database) -> None:
    raw_df = load_current_barra_files()
    clean_df = clean_barra_returns(raw_df)

    years = clean_df.select(pl.col("date").dt.year().unique().sort().alias("year"))[
        "year"
    ]

    for year in tqdm(years, desc="Daily Barra Returns"):
        year_df = clean_df.filter(pl.col("date").dt.year().eq(year))

        database.barra_returns_table.create_if_not_exists(year)
        database.barra_returns_table.upsert(year, year_df)

if __name__ == '__main__':
    from utils.enums import DatabaseName
    start = date(1995, 1, 1)
    end = date(2025, 12, 31)
    db = Database(DatabaseName.DEVELOPMENT)
    barra_returns_history_flow(start, end, db)
    barra_returns_daily_flow(db)