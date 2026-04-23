"""Compatibility helpers for accepting pandas or Polars DataFrames as input."""

from pandas import DataFrame


def _coerce_to_pandas(df) -> DataFrame:
    """Return a pandas DataFrame, converting from Polars if necessary.

    Args:
        df: A pandas or Polars DataFrame.

    Returns:
        A pandas DataFrame.

    Raises:
        TypeError: If ``df`` is neither a pandas nor a Polars DataFrame.
    """
    if isinstance(df, DataFrame):
        return df
    cls = type(df)
    if cls.__module__.startswith("polars"):
        # Convert column-by-column via Python lists to avoid a pyarrow dependency.
        import pandas as pd

        return pd.DataFrame({col: df[col].to_list() for col in df.columns})
    raise TypeError(f"Expected a pandas or Polars DataFrame, got {cls.__qualname__}")


def _is_dataframe(x) -> bool:
    """Return True if ``x`` is a pandas or Polars DataFrame."""
    if isinstance(x, DataFrame):
        return True
    cls = type(x)
    return cls.__module__.startswith("polars") and cls.__name__ == "DataFrame"
