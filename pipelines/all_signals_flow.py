from datetime import date
import click
from pipelines.barra_signals_flow import barra_signals_flow
from pipelines.ten_k_signals_flow import ten_k_signals_flow
from pipelines.utils.tables import Database


def all_signals_flow(start_date: date, end_date: date, database: Database) -> None:
    barra_signals_flow(start_date, end_date, database)
    ten_k_signals_flow(start_date, end_date, database)