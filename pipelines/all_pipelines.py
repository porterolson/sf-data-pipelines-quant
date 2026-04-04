from pipelines.barra_asset_ids_flow import barra_assets_daily_flow
from pipelines.barra_covariances_flow import (
    barra_covariances_daily_flow,
    barra_covariances_history_flow,
)
from pipelines.barra_exposures_flow import (
    barra_exposures_daily_flow,
    barra_exposures_history_flow,
)
from pipelines.barra_ids_flow import barra_asset_ids_daily_flow
from pipelines.barra_returns_flow import barra_returns_daily_flow, barra_returns_history_flow
from pipelines.barra_risk_flow import barra_risk_daily_flow, barra_risk_history_flow
from pipelines.ftse_russell_flow import ftse_russell_backfill_flow
from pipelines.barra_specific_returns import (
    barra_specific_returns_daily_flow,
    barra_specific_returns_history_flow,
)
from pipelines.barra_volume_flow import barra_volume_history_flow, barra_volume_daily_flow
from pipelines.assets_flow import assets_backfill_flow
from pipelines.crsp_daily_flow import crsp_daily_backfill_flow
from pipelines.crsp_monthly_flow import crsp_monthly_backfill_flow
from pipelines.crsp_events_flow import crsp_events_backfill_flow
from pipelines.crsp_v2_daily_flow import crsp_v2_daily_backfill_flow
from pipelines.crsp_v2_monthly_flow import crsp_v2_monthly_backfill_flow
from pipelines.barra_factors_flow import barra_factors_daily_flow
from pipelines.compustat_cik_flow import compustat_cik_backfill_flow
from pipelines.fama_french_flow import fama_french_5_factors_flow
import datetime as dt
from pipelines.utils.tables import Database


def barra_daily_flow(database: Database) -> None:
    # Assets table
    barra_returns_daily_flow(database)
    barra_specific_returns_daily_flow(database)
    barra_risk_daily_flow(database)
    barra_volume_daily_flow(database)

    # Covariance Matrix Components
    barra_exposures_daily_flow(database)
    barra_covariances_daily_flow(database)

    # Factors
    barra_factors_daily_flow(database)


def barra_history_flow(
    start_date: dt.date, end_date: dt.date, database: Database
) -> None:
    # Assets table
    barra_returns_history_flow(start_date, end_date, database)
    barra_specific_returns_history_flow(start_date, end_date, database)
    barra_risk_history_flow(start_date, end_date, database)
    barra_volume_history_flow(start_date, end_date, database)
    assets_backfill_flow(start_date, end_date, database)

    # Covariance Matrix Components
    barra_exposures_history_flow(start_date, end_date, database)
    barra_covariances_history_flow(start_date, end_date, database)


def id_mappings_flow(database: Database) -> None:
    barra_asset_ids_daily_flow(database)
    barra_assets_daily_flow(database)


def ftse_history_flow(
    start_date: dt.date, end_date: dt.date, database: Database, user: str
) -> None:
    """Note: requires logging in to WRDS when running."""
    ftse_russell_backfill_flow(start_date, end_date, database)


def crsp_history_flow(
    start_date: dt.date, end_date: dt.date, database: Database, user: str
) -> None:
    """Note: requires logging in to WRDS when running."""
    crsp_events_backfill_flow(start_date, end_date, database, user)
    crsp_monthly_backfill_flow(start_date, end_date, database, user)
    crsp_daily_backfill_flow(start_date, end_date, database, user)

def crsp_v2_history_flow(
    start_date: dt.date, end_date: dt.date, database: Database, user: str
) -> None:
    """Note: requires logging in to WRDS when running."""
    crsp_v2_daily_backfill_flow(start_date, end_date, database, user)
    crsp_v2_monthly_backfill_flow(start_date, end_date, database, user)


def barra_daily_pipeline(database: Database) -> None:
    barra_daily_flow(database)
    id_mappings_flow(database)
    assets_backfill_flow(dt.date(1995, 7, 31), dt.date.today(), database)


def barra_backfill_pipeline(
    start_date: dt.date, end_date: dt.date, database: Database
) -> None:
    barra_history_flow(start_date, end_date, database)
    id_mappings_flow(database)
    assets_backfill_flow(start_date, end_date, database)


def ftse_backfill_pipeline(
    start_date: dt.date, end_date: dt.date, database: Database, user: str
) -> None:
    ftse_history_flow(start_date, end_date, database, user)


def compustat_cik_backfill_pipeline(
    start_date: dt.date, end_date: dt.date, database: Database
) -> None:
    compustat_cik_backfill_flow(start_date, end_date, database)


def crsp_backfill_pipeline(
    start_date: dt.date, end_date: dt.date, database: Database, user: str
) -> None:
    crsp_history_flow(start_date, end_date, database, user)

def crsp_v2_backfill_pipeline(
    start_date: dt.date, end_date: dt.date, database: Database, user: str
) -> None:
    crsp_v2_history_flow(start_date, end_date, database, user)
