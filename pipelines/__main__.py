import click
import datetime as dt
from pipelines.all_pipelines import (
    barra_backfill_pipeline,
    ftse_backfill_pipeline,
    compustat_cik_backfill_pipeline,
    crsp_backfill_pipeline,
    crsp_v2_backfill_pipeline,
    barra_daily_pipeline,
    fama_french_5_factors_flow,
)
from pipelines.ten_k_filings_flow import ten_k_filings_flow
from pipelines.signals_flow import signals_flow
from pipelines.utils.enums import DatabaseName
from pipelines.utils.tables import Database

from dotenv import load_dotenv
import os

# Valid options
VALID_DATABASES = ["research", "production", "development"]
PIPELINE_TYPES = ["backfill", "update"]


@click.group()
def cli():
    """Main CLI entrypoint."""
    pass


@cli.command()
@click.argument(
    "pipeline_type", type=click.Choice(PIPELINE_TYPES, case_sensitive=False)
)
@click.option(
    "--database",
    type=click.Choice(VALID_DATABASES, case_sensitive=False),
    required=True,
    help="Target database (research or database).",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date(1995, 7, 31)),
    show_default=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date.today()),
    show_default=True,
    help="End date (YYYY-MM-DD).",
)
def barra(pipeline_type, database, start, end):
    match pipeline_type:
        case "backfill":
            start = start.date() if hasattr(start, "date") else start
            end = end.date() if hasattr(end, "date") else end

            click.echo(f"Running barra backfill on '{database}' from {start} to {end}.")

            database_name = DatabaseName(database)
            database_instance = Database(database_name)

            barra_backfill_pipeline(start, end, database_instance)

        case "update":
            click.echo(f"Running update for {database} database.")

            database_name = DatabaseName(database)
            database_instance = Database(database_name)

            barra_daily_pipeline(database_instance)


@cli.command()
@click.argument(
    "pipeline_type",
    type=click.Choice(
        ["backfill"], case_sensitive=False
    ),  # Update is currently not supported
)
@click.option(
    "--database",
    type=click.Choice(VALID_DATABASES, case_sensitive=False),
    required=True,
    help="Target database (research or database).",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date(1925, 1, 1)),
    show_default=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date.today()),
    show_default=True,
    help="End date (YYYY-MM-DD).",
)
def crsp(pipeline_type, database, start, end):
    load_dotenv(override=True)
    
    user = os.getenv("WRDS_USER")
    if user is None:
        raise EnvironmentError(
            "Missing required environment variable: WRDS_USER. "
            "Check your .env file."
        )

    match pipeline_type:
        case "backfill":
            start = start.date() if hasattr(start, "date") else start
            end = end.date() if hasattr(end, "date") else end

            click.echo(f"Running crsp backfill on '{database}' from {start} to {end}.")

            database_name = DatabaseName(database)
            database_instance = Database(database_name)

            crsp_backfill_pipeline(start, end, database_instance, user)


@cli.command()
@click.argument(
    "pipeline_type",
    type=click.Choice(
        ["backfill"], case_sensitive=False
    ),  # Update is currently not supported
)
@click.option(
    "--database",
    type=click.Choice(VALID_DATABASES, case_sensitive=False),
    required=True,
    help="Target database (research or database).",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date(1925, 1, 1)),
    show_default=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date.today()),
    show_default=True,
    help="End date (YYYY-MM-DD).",
)
def crsp_v2(pipeline_type, database, start, end):
    load_dotenv(override=True)
    
    user = os.getenv("WRDS_USER")
    if user is None:
        raise EnvironmentError(
            "Missing required environment variable: WRDS_USER. "
            "Check your .env file."
        )

    match pipeline_type:
        case "backfill":
            start = start.date() if hasattr(start, "date") else start
            end = end.date() if hasattr(end, "date") else end

            click.echo(f"Running crsp backfill on '{database}' from {start} to {end}.")

            database_name = DatabaseName(database)
            database_instance = Database(database_name)

            crsp_v2_backfill_pipeline(start, end, database_instance, user)

@cli.command()
@click.argument(
    "pipeline_type",
    type=click.Choice(
        ["backfill"], case_sensitive=False
    ),  # Update is currently not supported
)
@click.option(
    "--database",
    type=click.Choice(VALID_DATABASES, case_sensitive=False),
    required=True,
    help="Target database (research or database).",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date(1995, 7, 31)),
    show_default=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date.today()),
    show_default=True,
    help="End date (YYYY-MM-DD).",
)
def ftse(pipeline_type, database, start, end):
    load_dotenv(override=True)

    user = os.getenv("WRDS_USER")
    if user is None:
        raise EnvironmentError(
            "Missing required environment variable: WRDS_USER. "
            "Check your .env file."
        )

    match pipeline_type:
        case "backfill":
            start = start.date() if hasattr(start, "date") else start
            end = end.date() if hasattr(end, "date") else end

            click.echo(f"Running ftse backfill on '{database}' from {start} to {end}.")

            database_name = DatabaseName(database)
            database_instance = Database(database_name)

            ftse_backfill_pipeline(start, end, database_instance, user)


@cli.command(name="compustat-cik")
@click.argument(
    "pipeline_type",
    type=click.Choice(
        ["backfill"], case_sensitive=False
    ),
)
@click.option(
    "--database",
    type=click.Choice(VALID_DATABASES, case_sensitive=False),
    required=True,
    help="Target database (research or database).",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date(1995, 7, 31)),
    show_default=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date.today()),
    show_default=True,
    help="End date (YYYY-MM-DD).",
)
def compustat_cik(pipeline_type, database, start, end):
    load_dotenv(override=True)

    user = os.getenv("WRDS_USER")
    if user is None:
        raise EnvironmentError(
            "Missing required environment variable: WRDS_USER. "
            "Check your .env file."
        )

    match pipeline_type:
        case "backfill":
            start = start.date() if hasattr(start, "date") else start
            end = end.date() if hasattr(end, "date") else end

            click.echo(
                f"Running compustat-cik backfill on '{database}' from {start} to {end}."
            )

            database_name = DatabaseName(database)
            database_instance = Database(database_name)

            compustat_cik_backfill_pipeline(start, end, database_instance)


@cli.command()
@click.option(
    "--database",
    type=click.Choice(VALID_DATABASES, case_sensitive=False),
    required=True,
    help="Target database (research, production, or development).",
)
def fama_french(database):
    click.echo("Running Fama-French 5-factors pipeline...")

    database_name = DatabaseName(database)
    database_instance = Database(database_name)

    fama_french_5_factors_flow(database_instance)
    click.echo("Fama-French pipeline completed successfully.")


@cli.command()
@click.argument(
    "pipeline_type",
    type=click.Choice(
        ["backfill"], case_sensitive=False
    ),  # Update is currently not supported
)
@click.option(
    "--database",
    type=click.Choice(VALID_DATABASES, case_sensitive=False),
    required=True,
    help="Target database (research or database).",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date(1996, 7, 31)),
    show_default=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date.today()),
    show_default=True,
    help="End date (YYYY-MM-DD).",
)
def signals(pipeline_type, database, start, end):
    match pipeline_type:
        case "backfill":
            start = start.date() if hasattr(start, "date") else start
            end = end.date() if hasattr(end, "date") else end

            click.echo(f"Running signals backfill on '{database}' from {start} to {end}...")

            database_name = DatabaseName(database)
            database_instance = Database(database_name)

            signals_flow(start, end, database_instance)


@cli.command(name="ten-k")
@click.argument(
    "pipeline_type",
    type=click.Choice(
        ["backfill"], case_sensitive=False
    ),
)
@click.option(
    "--database",
    type=click.Choice(VALID_DATABASES, case_sensitive=False),
    required=True,
    help="Target database (research or database).",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date(1995, 7, 31)),
    show_default=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(dt.date.today()),
    show_default=True,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--today",
    "today_mode",
    is_flag=True,
    help="Use the latest FTSE Russell snapshot on or before today and only recheck companies whose last saved 10-K is at least 245 trading days old.",
)
def ten_k(pipeline_type, database, start, end, today_mode):
    load_dotenv(override=True)

    match pipeline_type:
        case "backfill":
            if today_mode:
                today = dt.date.today()
                click.echo(
                    f"Running 10-K daily update on '{database}' using the latest FTSE Russell snapshot on or before {today}."
                )
            else:
                start = start.date() if hasattr(start, "date") else start
                end = end.date() if hasattr(end, "date") else end

                click.echo(
                    f"Running 10-K backfill on '{database}' from {start} to {end}."
                )

            database_name = DatabaseName(database)
            database_instance = Database(database_name)

            if today_mode:
                ten_k_filings_flow(dt.date.today(), dt.date.today(), database_instance, today_mode=True)
            else:
                ten_k_filings_flow(start, end, database_instance)


if __name__ == "__main__":
    cli()
