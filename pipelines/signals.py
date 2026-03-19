import polars as pl


def zscore_scorer(df: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional z-score per date."""
    return df.with_columns(
        pl.col("signal_value")
        .sub(pl.col("signal_value").mean())
        .truediv(pl.col("signal_value").std())
        .over("date")
        .alias("score")
    )


def ic_alphatizer(df: pl.DataFrame, ic: float = 0.05) -> pl.DataFrame:
    """Alpha = score * IC * specific_risk."""
    return df.with_columns(
        pl.col("score").mul(ic).mul(pl.col("specific_risk")).alias("alpha")
    )


def momentum() -> dict:
    return {
        "expr": (
            pl.col("return")
            .log1p()
            .rolling_sum(window_size=230)
            .shift(22)
            .over("barrid")
            .alias("momentum")
        ),
        "scorer": zscore_scorer,
        "alphatizer": ic_alphatizer,
    }


def reversal() -> dict:
    return {
        "expr": (
            pl.col("return")
            .log1p()
            .rolling_sum(window_size=22)
            .mul(-1)
            .over("barrid")
            .alias("reversal")
        ),
        "scorer": zscore_scorer,
        "alphatizer": ic_alphatizer,
    }


def beta() -> dict:
    return {
        "expr": pl.col("predicted_beta").mul(-1).over("barrid").alias("beta"),
        "scorer": zscore_scorer,
        "alphatizer": ic_alphatizer,
    }


# Registry for easy lookup
SIGNALS = {
    "momentum": momentum(),
    "reversal": reversal(),
    "beta": beta(),
}