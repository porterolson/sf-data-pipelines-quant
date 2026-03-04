import urllib.request
import zipfile
from pathlib import Path
import polars as pl
import io
from pipelines.utils.tables import Database
from pipelines.utils.enums import DatabaseName


def fama_french_5_factors_flow(database: Database) -> None:
    """
    Download and process Fama-French 5-factor daily data.
    Saves to the database using the fama_french_factors_table.
    """
    # URL for the Fama-French 5 factors (daily data)
    ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

    # Download the zip file
    urllib.request.urlretrieve(ff_url, "fama_french.zip")

    # Extract the CSV file
    with zipfile.ZipFile("fama_french.zip", "r") as zip_file:
        zip_file.extractall()

    # Read the CSV file with polars
    # Skip first 4 rows (header info) and read only the daily data
    with open("F-F_Research_Data_5_Factors_2x3_daily.csv", "r") as f:
        lines = f.readlines()

    # Find where daily data ends (when we hit empty line or "Annual" section)
    daily_end_idx = None
    for i, line in enumerate(lines[4:], start=4):  # Start after header rows
        if line.strip() == "" or "Annual" in line:
            daily_end_idx = i
            break

    # Read only the daily data section
    csv_data = "".join(lines[3:daily_end_idx])

    # Parse with polars
    data = pl.read_csv(
        io.StringIO(csv_data),
        has_header=True,
        new_columns=["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"],
    ).with_columns(
        [
            # Convert date format (YYYYMMDD) to actual date
            pl.col("date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d"),
            # Convert from percentage to decimal
            pl.col("mkt_rf").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("smb").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("hml").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("rmw").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("cma").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("rf").str.strip_chars().cast(pl.Float64).truediv(100),
        ]
    )

    # Save to database using the table interface
    table = database.fama_french_table
    table.overwrite(data)

    # Clean up downloaded files
    Path("fama_french.zip").unlink(missing_ok=True)
    Path("F-F_Research_Data_5_Factors_2x3_daily.csv").unlink(missing_ok=True)
