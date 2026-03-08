import polars as pl
import os
from dotenv import load_dotenv
from pipelines.utils.factors import factors
from typing import Optional
from pipelines.utils.enums import DatabaseName


class Table:
    def __init__(
        self,
        database: DatabaseName,
        name: str,
        schema: dict[str, pl.DataType],
        ids=list[str],
    ) -> None:
        load_dotenv(override=True)
        home, user = os.getenv("ROOT").split("/")[1:3]
        self._base_path = f"/{home}/{user}/groups/grp_quant/database/{database.value}"

        self._name = name
        self._schema = schema
        self._ids = ids

        os.makedirs(f"{self._base_path}/{self._name}", exist_ok=True)

    def _file_path(self, year: int | None = None) -> str:
        if year is not None:
            return f"{self._base_path}/{self._name}/{self._name}_{year}.parquet"
        else:
            return f"{self._base_path}/{self._name}/{self._name}_*.parquet"

    def exists(self, year: int) -> bool:
        return os.path.exists(self._file_path(year))

    def create_if_not_exists(self, year: int) -> None:
        if not os.path.exists(self._file_path(year)):
            pl.DataFrame(schema=self._schema).write_parquet(self._file_path(year))

    def read(self, year: int | None = None) -> pl.LazyFrame:
        if year is None:
            return pl.scan_parquet(self._file_path())
        else:
            return pl.scan_parquet(self._file_path(year))

    def read_id_file(self) -> pl.LazyFrame:
        return pl.scan_parquet(f"{self._base_path}/{self._name}/{self._name}.parquet")
    
    def overwrite(self, df: pl.DataFrame) -> None:
        df.write_parquet(f"{self._base_path}/{self._name}/{self._name}.parquet")

    def upsert(self, year: int, rows: pl.DataFrame) -> None:
        (
            pl.scan_parquet(self._file_path(year))
            .update(rows.lazy(), on=self._ids, how="full")
            .collect()
            .write_parquet(self._file_path(year))
        )

    def update(
        self, year: int, rows: pl.DataFrame, on: Optional[list[str]] = None
    ) -> None:
        on = on or self._ids
        (
            pl.scan_parquet(self._file_path(year))
            .update(rows.lazy(), on=on, how="left")
            .collect()
            .write_parquet(self._file_path(year))
        )

    def delete(self, year: int) -> None:
        """Delete parquet file for a specific year."""
        file_path = self._file_path(year)
        if os.path.exists(file_path):
            os.remove(file_path)

    def update_asof(
        self,
        year: int,
        right_df: pl.DataFrame,
        left_on: str,
        right_on: str,
        by: str | list[str],
        strategy: str = 'backward',
        drop_right_cols: list[str] | None = None
    ) -> pl.DataFrame:
        """
        Perform join_asof with intelligent column conflict handling.
        
        Overlapping columns are coalesced (preferring right_df values).
        Columns unique to either DataFrame are preserved.
        
        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame to join
            left_on: Temporal column in left_df
            right_on: Temporal column in right_df
            by: Column(s) to join on
            strategy: Join strategy ('backward', 'forward', 'nearest')
            drop_right_cols: Columns from right_df to exclude from result
        
        Returns:
            Joined DataFrame with resolved column conflicts
        """
        left_df = pl.scan_parquet(self._file_path(year))

        if drop_right_cols is None:
            drop_right_cols = []
        
        on = by if isinstance(by, list) else [by]
        on_with_left = on + [left_on]
        
        left_cols = set(left_df.collect_schema().names())
        right_cols = set(right_df.collect_schema().names())
        
        right_cols_to_keep = right_cols - set(drop_right_cols) - {right_on}
        
        # Columns in both DataFrames get suffix '_updated' during join
        overlap_cols = (left_cols & right_cols_to_keep) - set(on_with_left)
        
        # Columns unique to each DataFrame
        left_only_cols = left_cols - right_cols_to_keep - set(on_with_left)
        right_only_cols = right_cols_to_keep - left_cols - set(on)

        joined = (
            left_df
            .sort(*on, left_on)
            .join_asof(
                other=right_df.lazy().sort(*on, right_on),
                left_on=left_on,
                right_on=right_on,
                by=by,
                strategy=strategy,
                suffix='_updated',
                check_sortedness=False
            )
        )
        
        select_exprs = on_with_left.copy()
        
        # Coalesce overlapping columns (prefer right_df)
        for col in sorted(overlap_cols):
            select_exprs.append(pl.coalesce(f"{col}_updated", col).alias(col))
        
        # Keep left-only columns
        for col in sorted(left_only_cols):
            select_exprs.append(pl.col(col))
        
        # Keep right-only columns (no suffix added by join)
        for col in sorted(right_only_cols):
            select_exprs.append(pl.col(col))

        joined.select(select_exprs).collect().write_parquet(self._file_path(year))
        

class Database:
    def __init__(self, database_name: DatabaseName):
        self._database_name = database_name

    @property
    def assets_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="assets",
            schema={
                "date": pl.Date,
                "rootid": pl.String,
                "barrid": pl.String,
                "issuerid": pl.String,
                "instrument": pl.String,
                "name": pl.String,
                "cusip": pl.String,
                "isin": pl.String,
                "cisn": pl.String,
                "ticker": pl.String,
                "price": pl.Float64,
                "return": pl.Float64,
                "specific_return": pl.Float64,
                "market_cap": pl.Float64,
                "price_source": pl.String,
                "currency": pl.String,
                "iso_country_code": pl.String,
                "iso_currency_code": pl.String,
                "yield": pl.Float64,
                "total_risk": pl.Float64,
                "specific_risk": pl.Float64,
                "historical_beta": pl.Float64,
                "predicted_beta": pl.Float64,
                "russell_1000": pl.Boolean,
                "russell_2000": pl.Boolean,
                "in_universe": pl.Boolean,
                "daily_volume": pl.Float64,
                "average_daily_volume_30": pl.Float64,
                "average_daily_volume_60": pl.Float64,
                "average_daily_volume_90": pl.Float64,
                "bid_ask_spread": pl.Float64,
                "average_daily_bid_ask_spread_30": pl.Float64,
                "average_daily_bid_ask_spread_60": pl.Float64,
                "average_daily_bid_ask_spread_90": pl.Float64,
            },
            ids=["date", "barrid"],
        )

    @property
    def barra_returns_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="barra_returns",
            schema={
                "barrid": pl.String,
                "price": pl.Float64,
                "market_cap": pl.Float64,
                "price_source": pl.String,
                "currency": pl.String,
                "return": pl.Float64,
                "date": pl.Date,
            },
            ids=["date", "barrid"],
        )

    @property
    def barra_specific_returns_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="barra_specific_returns",
            schema={
                "barrid": pl.String,
                "specific_return": pl.Float64,
                "date": pl.Date,
            },
            ids=["date", "barrid"],
        )

    @property
    def barra_risk_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="barra_risk",
            schema={
                "barrid": pl.String,
                "yield": pl.Float64,
                "total_risk": pl.Float64,
                "specific_risk": pl.Float64,
                "historical_beta": pl.Float64,
                "predicted_beta": pl.Float64,
                "date": pl.Date,
            },
            ids=["date", "barrid"],
        )

    @property
    def barra_volume_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="barra_volume",
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                "daily_volume": pl.Float64,
                "average_daily_volume_30": pl.Float64,
                "average_daily_volume_60": pl.Float64,
                "average_daily_volume_90": pl.Float64,
                "bid_ask_spread": pl.Float64,
                "average_daily_bid_ask_spread_30": pl.Float64,
                "average_daily_bid_ask_spread_60": pl.Float64,
                "average_daily_bid_ask_spread_90": pl.Float64,
            },
            ids=["date", "barrid"],
        )

    @property
    def exposures_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="exposures",
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                **{factor: pl.Float64 for factor in factors},
            },
            ids=["date", "barrid"],
        )

    @property
    def covariances_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="covariances",
            schema={
                "date": pl.Date,
                "factor_1": pl.String,
                **{factor: pl.Float64 for factor in factors},
            },
            ids=["date", "factor_1"],
        )

    @property
    def crsp_events_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="crsp_events",
            schema={
                "date": pl.Date,
                "permno": pl.Int64,
                "ticker": pl.String,
                "shrcd": pl.Int64,
                "exchcd": pl.Int64,
            },
            ids=["date", "permno"],
        )

    @property
    def crsp_monthly_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="crsp_monthly",
            schema={
                "date": pl.Date,
                "permno": pl.Int64,
                "cusip": pl.String,
                "ret": pl.Float64,
                "retx": pl.Float64,
                "prc": pl.Float64,
                "vol": pl.Int64,
                "shrout": pl.Int64,
            },
            ids=["date", "permno"],
        )

    @property
    def crsp_daily_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="crsp_daily",
            schema={
                "date": pl.Date,
                "permno": pl.Int64,
                "cusip": pl.String,
                "ret": pl.Float64,
                "retx": pl.Float64,
                "prc": pl.Float64,
                "vol": pl.Int64,
                "openprc": pl.Float64,
                "askhi": pl.Float64,
                "bidlo": pl.Float64,
                "shrout": pl.Int64,
            },
            ids=["date", "permno"],
        )
    
    @property
    def crsp_v2_monthly_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="crsp_v2_monthly",
            schema={
                "date": pl.Date,
                "permno": pl.Int64,
                "ticker": pl.String,
                "cusip": pl.String,
                "ret": pl.Float64,
                "retx": pl.Float64,
                "prc": pl.Float64,
                "vol": pl.Int64,
                "shrout": pl.Int64,
                "primaryexch": pl.String,
                "securitytype": pl.String
            },
            ids=["date", "permno"],
        )
    
    @property
    def crsp_v2_daily_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="crsp_v2_daily",
            schema={
                "date": pl.Date,
                "permno": pl.Int64,
                "cusip": pl.String,
                "ticker": pl.String,
                "ret": pl.Float64,
                "retx": pl.Float64,
                "prc": pl.Float64,
                "vol": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "shrout": pl.Int64,
                "primaryexch": pl.String,
                "securitytype": pl.String
            },
            ids=["date", "permno"],
        )

    @property
    def factors_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="factors",
            schema={
                "date": pl.Date,
                **{factor: pl.Float64 for factor in factors},
            },
            ids=["date"],
        )

    @property
    def signals_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="signals",
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                "signal_name": pl.String,
                "signal_value": pl.Float64,
            },
            ids=["date", "barrid", "signal_name"],
        )

    @property
    def active_weights_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="active_weights",
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                "signal": pl.String,
                "weight": pl.Float64,
            },
            ids=["date", "barrid", "signal"],
        )

    @property
    def composite_alphas_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="composite_alphas",
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                "name": pl.String,
                "alpha": pl.Float64,
            },
            ids=["date", "barrid", "name"],
        )

    @property
    def asset_ids_table(self) -> Table:
        return Table(
            database=self._database_name,
            name ="asset_ids",
            schema={
                "start_date": pl.Date,
                "end_date": pl.Date,
                "rootid": pl.String,
                "barrid": pl.String,
                "issuerid": pl.String,
                "instrument": pl.String,
                "name": pl.String,
                "iso_country_code": pl.String,
                "iso_currency_code": pl.String,
            },
            ids=['barrid','start_date', 'rootid', 'end_date']
        )

    @property
    def barra_ids_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="barra_ids",
            schema={
                "barrid": pl.String,
                "asset_id_type": pl.String,
                "asset_id": pl.String,
                "start_date": pl.Date,
                "end_date": pl.Date,
            },
            ids=['barrid','start_date', "asset_id_type", "end_date"]
        )

    @property
    def ftse_russell_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="ftse_russell",
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                "cusip": pl.String,
                "russell_1000": pl.Boolean,
                "russell_2000": pl.Boolean,
                "in_universe": pl.Boolean,
            },
            ids=["date", "barrid"],
        )

    @property
    def fama_french_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="fama_french",
            schema={
                "date": pl.Date,
                "mkt_rf": pl.Float64,
                "smb": pl.Float64,
                "hml": pl.Float64,
                "rmw": pl.Float64,
                "cma": pl.Float64,
                "rf": pl.Float64,
            },
            ids=["date"],
        )

    @property
    def ftse_russell_table(self) -> Table:
        return Table(
            database=self._database_name,
            name="ftse_russell",
            schema={
                "date": pl.Date,
                "cusip": pl.String,
                "russell_2000": pl.Boolean,
                "russell_1000": pl.Boolean,
            },
            ids=["date", "cusip"],
        )
    