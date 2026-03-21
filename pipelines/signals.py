import polars as pl

# Z-Score functions
def zscore_scorer(df: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional z-score per date."""
    return df.with_columns(
        pl.col("signal_value")
        .sub(pl.col("signal_value").mean())
        .truediv(pl.col("signal_value").std())
        .over("date")
        .alias("score")
    )

def win_zscore(df: pl.DataFrame) -> pl.DataFrame:
    return (df
            .with_columns(
                pl.col("signal_value")
                .sub(pl.col("signal_value").mean())
                .truediv(pl.col("signal_value").std())
                .over("date")
                .alias("score")
                .clip(lower_bound=-2.0, upper_bound=2.0)
            )
            .sort(["barrid", "date"])
            .with_columns(dollar_volume=pl.col("daily_volume").mul(pl.col("price")).log1p())
            .with_columns(
                dollar_volume_mean=pl.col("dollar_volume")
                .rolling_mean(window_size=252, min_samples=1)
                .over("barrid"),
                dollar_volume_std=pl.col("dollar_volume")
                .rolling_std(window_size=252, min_samples=2)
                .over("barrid"),
            )
            .with_columns(
                volume_score=(
                    (pl.col("dollar_volume") - pl.col("dollar_volume_mean"))
                    /
                    pl.col("dollar_volume_std").fill_null(1.0).clip(lower_bound=0.0001)
                )
                .fill_null(0.0)
                .alias("volume_score")
            )
        )


# alpha functions
def ic_alphatizer(df: pl.DataFrame, ic: float = 0.05) -> pl.DataFrame:
    """Alpha = score * IC * specific_risk."""
    return df.with_columns(
        pl.col("score").mul(ic).mul(pl.col("specific_risk")).alias("alpha")
    )

def gk_alpha(df: pl.DataFrame, ic: float = 0.05) -> pl.DataFrame:
    return (df
            .with_columns(
                gk_alpha=pl.col("score") * ic * pl.col("specific_risk")
            )
            .with_columns(
                alpha=pl.when((pl.col("score").eq(2.0)) & (pl.col("volume_score").ge(2.0)))
                .then(0.0)
                .otherwise(pl.col("gk_alpha"))
            )
            .drop('gk_alpha')
            .sort("date", "barrid")
    )


# Signals
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
        "expr": (
                pl.col("predicted_beta")
                .mul(-1)
                .over("barrid")
                .alias("beta")
            ),
        "scorer": zscore_scorer,
        "alphatizer": ic_alphatizer,
    }

def barra_reversal() -> dict:
    return {
        "expr": (
                pl.col("specific_return")
                .ewm_mean(span=5, min_samples=5)
                .mul(-1)
                .shift(1)
                .over("barrid")
                .alias("barra_reversal")
            ),
        "scorer": win_zscore,
        "alphatizer": gk_alpha,
     }

def barra_momentum() -> dict:
    return {
        "expr": (
                pl.col("specific_return")
                .log1p()
                .rolling_sum(230)
                .truediv(pl.col("specific_return").rolling_std(230))
                .shift(21)
                .over("barrid")
                .alias("barra_momentum")
            ),
        "scorer": zscore_scorer,
        "alphatizer": ic_alphatizer,
    }

def ivol() -> dict:
    return {
        "expr": (
                pl.col("specific_risk")
                .mul(-1)
                .shift(1)
                .over("barrid")
                .alias("ivol")
            ),
        "scorer": zscore_scorer,
        "alphatizer": ic_alphatizer,
    }

# Registry for easy lookup
SIGNALS = {
    "momentum": momentum(),
    "reversal": reversal(),
    "beta": beta(),
    "barra_reversal": barra_reversal(),
    "barra_momentum": barra_momentum(),
    "ivol": ivol(),
}